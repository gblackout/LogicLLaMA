import json
from os.path import join as joinpath
import os
import re
import nltk
from copy import deepcopy
import numpy as np
from nltk.tree import Tree
from itertools import product, combinations, permutations
from typing import Optional, Union, List, Dict, Callable
from utils import all_exists, has_same_obj_in_list, DataPreparer
import random
from Levenshtein import distance as edit_dist


# TODO inefficient
op_ls = ['⊕', '∨', '∧', '→', '↔', '∀', '∃', '¬', '(', ')', ',']

sym_reg = re.compile(r'[^⊕∨∧→↔∀∃¬(),]+')

cfg_template = """
S -> F | Q F
Q -> QUANT VAR | QUANT VAR Q
F -> '¬' '(' F ')' | '(' F ')' | F OP F | L
OP -> '⊕' | '∨' | '∧' | '→' | '↔'
L -> '¬' PRED '(' TERMS ')' | PRED '(' TERMS ')'
TERMS -> TERM | TERM ',' TERMS
TERM -> CONST | VAR
QUANT -> '∀' | '∃'
"""

# used in perturbation
last_nt_nodes = set(['PRED', 'OP', 'CONST', 'VAR', 'QUANT'])
# used in node insertion
insertable_nt_nodes = set(['Q', 'S', 'TERMS', 'F'])
# used in node deletion
deleteable_nt_nodes = set(['Q', 'TERMS', 'F', 'L'])

def parse_text_FOL_to_tree(rule_str):
    """
        Parse a text FOL rule into nltk.tree
        
        Returns: nltk.tree, or None if the parse fails
    """
    rule_str = reorder_quantifiers(rule_str)
    
    r, parsed_fol_str = msplit(rule_str)
    cfg_str = make_cfg_str(r)

    grammar = nltk.CFG.fromstring(cfg_str)
    parser = nltk.ChartParser(grammar)
    tree = parser.parse_one(r)
    
    return tree


def reorder_quantifiers(rule_str):
    matches = re.findall(r'[∃∀]\w', rule_str)
    for match in matches[::-1]:
        rule_str = '%s ' % match + rule_str.replace(match, '', 1)
    return rule_str


def msplit(s):
    for op in op_ls:
        s = s.replace(op, ' %s ' % op)
    r = [e.strip() for e in s.split()]
    #remove ' from the string if it contains any: this causes error in nltk cfg parsing
    r = [e.replace('\'', '') for e in r]
    r = [e for e in r if e != '']
    
    # deal with symbols with spaces like "dc universe" and turn it to "DcUniverse"
    res = []
    cur_str_ls = []
    for e in r:
        if (len(e) > 1) and sym_reg.match(e):            
            cur_str_ls.append(e[0].upper() + e[1:])            
        else:
            if len(cur_str_ls) > 0:
                res.extend([''.join(cur_str_ls), e])
            else:
                res.extend([e])
            cur_str_ls = []
    if len(cur_str_ls) > 0:
        res.append(''.join(cur_str_ls))
    
    # re-generate the FOL string
    make_str_ls = []
    for ind, e in enumerate(r):
        if re.match(r'[⊕∨∧→↔]', e):
            make_str_ls.append(' %s ' % e)
        elif re.match(r',', e):
            make_str_ls.append('%s ' % e)
        # a logical variable
        elif (len(e) == 1) and re.match(r'\w', e):
            if ((ind - 1) >= 0) and ((r[ind-1] == '∃') or (r[ind-1] == '∀')):
                make_str_ls.append('%s ' % e)
            else:
                make_str_ls.append(e)
        else:
            make_str_ls.append(e)
    
    return res, ''.join(make_str_ls)


def make_cfg_str(token_ls):
    """
    NOTE: since nltk does not support reg strs like \w+, we cannot separately recognize VAR, PRED, and CONST.
    Instead, we first allow VAR, PRED, and CONST to be matched with all symbols found in the FOL; once the tree is
    parsered, we then go back and figure out the exact type of each symbols
    """
    sym_ls = list(set([e for e in token_ls if sym_reg.match(e)]))
    sym_str = ' | '.join(["'%s'" % s for s in sym_ls])
    cfg_str = cfg_template + 'VAR -> %s\nPRED -> %s\nCONST -> %s' % (sym_str,sym_str,sym_str)
    return cfg_str


def symbol_resolution(tree):
    lvars, consts, preds = set(), set(), set()
    
    if tree[0].label() == 'Q':
        isFOL = True
        main_tree = tree[1]
        for sym, tag in tree[0].pos():
            if tag == 'VAR':
                lvars.add(sym)
    else:
        isFOL = False
        main_tree = tree[0]
        
    preorder_resolution(main_tree, lvars, consts, preds)
    
    return isFOL, lvars, consts, preds


def preorder_resolution(tree, lvars, consts, preds):
    # reached terminal nodes
    if isinstance(tree, str):
        return
    
    if tree.label() == 'PRED':
        preds.add(tree[0])
        return
    
    if tree.label() == 'TERM':
        sym = tree[0][0]
        if sym in lvars:
            tree[0].set_label('VAR')
        else:
            tree[0].set_label('CONST')
            consts.add(sym)
        return
    
    for child in tree:
        preorder_resolution(child, lvars, consts, preds)

        
class Rule:
    def __init__(self, isFOL, lvars, consts, preds, tree):
        self.isFOL = isFOL
        self.lvars = lvars
        self.consts = consts
        self.preds = preds
        self.tree = tree
        
    def rule_str(self):
        # TODO inefficient
        _, rule_str = msplit(''.join(self.tree.leaves()))
        return rule_str

    def _get_nodes(
            self,
            root,
            nodes,
            allowed_labels=None,
            not_allowed_nodes=None,
            not_allowed_nodes_in_subtree=False
    ):
        all_child_allowed = True

        if isinstance(root, str):
            return all_child_allowed

        # post-order check children
        for child in root:
            child_allowed = self._get_nodes(
                child,
                nodes,
                allowed_labels,
                not_allowed_nodes,
                not_allowed_nodes_in_subtree
            )
            all_child_allowed = all_child_allowed and child_allowed
        # this is twisted... what i mean is that if you don't want to filter the subtrees then simply set
        # all_child_allowed to true regardless of the post-order check result
        if not not_allowed_nodes_in_subtree:
            all_child_allowed = True

        root_is_allowed = (not_allowed_nodes is None) or all(e is not root for e in not_allowed_nodes)
        tree_is_allowed = all_child_allowed and root_is_allowed

        if (
                ((allowed_labels is None) or (root.label() in allowed_labels))
                and tree_is_allowed
        ):
            nodes.append(root)

        return tree_is_allowed

    def get_nodes(
            self,
            root,
            allowed_labels=None,
            not_allowed_nodes=None,
            not_allowed_nodes_in_subtree=False
    ):
        """
            get tree nodes from a tree

            Args:
                root:
                allowed_labels: None or a set of strs; only nodes with the allowed label is included
                not_allowed_nodes: None or a list of node objects; only node objects that are not in the list
                is included
                not_allowed_nodes_in_subtree: set to True to also filter out nodes whose subtree nodes are in
                not_allowed_nodes, this is used when finding the deletable nodes (we want to find nodes whose entire
                subtree is not perturbed before)
        """
        if not_allowed_nodes_in_subtree:
            assert all_exists(not_allowed_nodes), 'must specify not_allowed_nodes'

        nodes = []
        self._get_nodes(
            root,
            nodes,
            allowed_labels,
            not_allowed_nodes,
            not_allowed_nodes_in_subtree
        )
        return nodes
    
    def random_node_by_label(
            self,
            root,
            allowed_labels=None,
            not_allowed_nodes=None,
            not_allowed_nodes_in_subtree=False
    ):
        nodes = self.get_nodes(root, allowed_labels, not_allowed_nodes, not_allowed_nodes_in_subtree)
        choice = nodes[int(np.random.randint(len(nodes)))] if len(nodes) > 0 else None
        return choice
    
    def get_all_fopf(self, root, res):
        if isinstance(root, str):
            return
        
        if root.label() == 'F':
            if len(root) == 3 and all(not isinstance(child, str) for child in root):
                # this is a F - F OP F subtree
                res.append(root)                
            
        for child in root:
            self.get_all_fopf(child, res)
    
    def order_of(self, node):
        cnt, flag = self._inorder_order(self.tree, node, 0)
        assert flag
        return cnt

    def parent_of(self, root, node):
        if isinstance(root, str):
            return None

        for child in root:
            if child is node:
                return root
            parent = self.parent_of(child, node)
            if all_exists(parent):
                return parent

        return None
        
    def _inorder_order(self, root, node, order_cnt):
        if root is node:
            return order_cnt + 1, True
        
        if isinstance(root, str):
            return order_cnt + (root == node[0]), False
        
        cnt = order_cnt
        flag = False
        for n in root:
            cnt, flag = self._inorder_order(n, node, cnt)
            if flag:
                break
        
        return cnt, flag
        

class Sample:
    def __init__(self, nl, rule: Rule):
        self.nl = nl
        self.rule = rule
        self.perturbed_rule = None
        
        word_reg = re.compile(r'^\w+$')
        nl_words = [e.strip().lower() for e in self.nl.split()]
        nl_words = [e for e in nl_words if word_reg.match(e) is not None]
        
        self.symbols = list(self.rule.lvars) + list(self.rule.consts) + list(self.rule.preds) + nl_words
        assert len(self.symbols) > 1, 'to avoid dead loop; maybe improve later'

    @classmethod
    def get_changeable_nodes(cls, rule: Rule, occupied_nodes):
        return rule.get_nodes(
            rule.tree,
            allowed_labels=last_nt_nodes,
            not_allowed_nodes=occupied_nodes,
            not_allowed_nodes_in_subtree=False
        )

    @classmethod
    def get_insertable_nodes(cls, rule: Rule):
        return rule.get_nodes(
            rule.tree,
            allowed_labels=insertable_nt_nodes,
            not_allowed_nodes=None,
            not_allowed_nodes_in_subtree=False
        )

    @classmethod
    def get_deletable_nodes(cls, rule: Rule, occupied_nodes):

        # first, we find nodes that can be deleted entirely, this means its subtree cannot be occupied
        nodes = rule.get_nodes(
            rule.tree,
            allowed_labels=deleteable_nt_nodes,
            not_allowed_nodes=occupied_nodes,
            not_allowed_nodes_in_subtree=False
        )

        deletable_nodes = []
        for node in nodes:
            label = node.label()

            if label == 'Q':
                # a Q node whose direct QUANT and VAR children are not occupied is deletable
                if has_same_obj_in_list(node[0], occupied_nodes) or has_same_obj_in_list(node[1], occupied_nodes):
                    continue

            elif label == 'TERMS':
                # only TERMS node with a TERMS parent is deletable
                parent = rule.parent_of(rule.tree, node)
                if parent.label() != 'TERMS':
                    continue
                # only TERMS node with unoccupied direct TERM and VAR child is deletable
                if has_same_obj_in_list(node[0], occupied_nodes) or has_same_obj_in_list(node[0][0], occupied_nodes):
                    continue

            elif label == 'F':
                # only F node with siblings of F OP F is deletable
                parent = rule.parent_of(rule.tree, node)
                if len(parent) != 3:
                    continue
                if isinstance(parent[0], str):
                    continue
                if isinstance(parent[2], str):
                    continue
                if not (
                        parent[0].label() == 'F'
                        and parent[1].label() == 'OP'
                        and parent[2].label() == 'F'
                ):
                    continue
                # only F node with an entire unoccupied tree is deletable
                # I check this by checking if this F node is in the unoccupied_nodes while
                # not_allowed_nodes_in_subtree=True
                unoccupied_nodes = rule.get_nodes(
                    node,
                    allowed_labels=deleteable_nt_nodes,
                    not_allowed_nodes=occupied_nodes,
                    not_allowed_nodes_in_subtree=True
                )
                if not has_same_obj_in_list(node, unoccupied_nodes):
                    continue

            else:
                continue

            deletable_nodes.append(node)

        # secondly, we find nodes that has deletable negation, where it's fine if the subtree is occupied
        nodes_with_deletable_negation = []
        for node in nodes:
            label = node.label()
            # only F/L node has a deletable negation if it has negation in it
            if (label == 'F') or (label == 'L'):
                if isinstance(node[0], str) and (node[0] == '¬'):
                    nodes_with_deletable_negation.append(node)

        # combine the possible deletions and assign a mode tag
        deletable_nodes = \
            [(e, 'delete_node') for e in deletable_nodes] \
            + [(e, 'delete_negation') for e in nodes_with_deletable_negation]

        return deletable_nodes



class VecRuleEvaluator:

    dummy_input_str: str = '#DUMMY'
    dummy_distance: int = 10000

    @classmethod
    def default_input_similarity(cls, e1: str, e2: str):
        if e1.startswith(cls.dummy_input_str) or e2.startswith(cls.dummy_input_str):
            return cls.dummy_distance
        return edit_dist(e1, e2)

    @classmethod
    def enumerate_bindings_with_greedy_match(cls, ls1: List[str], ls2: List[str], top_n: int):
        """
            Given two lists of strings ls1 and ls2, yields the ind bindings of ls2 strings that matches the strings in
            ls1. I use greedy match and yields starts from the best to the worst binding until full enumeration or hit
            the top_n bound
        """

        used_inds = []

        def _enum_bindings(ind1: int):
            if ind1 == len(ls1):
                yield deepcopy(used_inds)
                return
            e1 = ls1[ind1]
            match_ls = [
                (ind, cls.default_input_similarity(e1, e2))
                for ind, e2 in enumerate(ls2) if ind not in used_inds
            ]
            match_ls.sort(key=lambda x:x[1])
            for ind, dist in match_ls:
                used_inds.append(ind)
                for inds in _enum_bindings(ind1+1):
                    yield inds
                used_inds.pop()

        for cnt, ind_ls in enumerate(_enum_bindings(0)):
            yield ind_ls
            if cnt + 1 == top_n:
                break

    @classmethod
    def find_inputs(cls, root, input_set=None):
        if isinstance(root, str):
            return

        label = root.label()
        
        if label == 'L':
            literal_str = ''.join(root.leaves())
            literal_str = literal_str[1:] if literal_str[0] == '¬' else literal_str
            if input_set is None:
                input_set = set()
            input_set.add(literal_str)
        else:
            for child in root:
                cls.find_inputs(child, input_set)

    @classmethod
    def gen_input_vecs(cls, num_inputs):
        return np.array(list(product([False, True], repeat=num_inputs)))
    
    
    @classmethod
    def from_nltk_tree(cls, root, name2ind_dict, input_vecs):
        assert not isinstance(root, str), 'something wrong with the rule or the algo; you should not parse a leave'
    
        label = root.label()
        
        if label == 'S':
            
            return cls.from_nltk_tree(root[-1], name2ind_dict, input_vecs)
        
        elif label == 'F':
            
            # the case F -> L            
            if (len(root) == 1) and (root[0].label() == 'L'):
                return cls.from_nltk_tree(root[0], name2ind_dict, input_vecs)
            
            # the case F -> '¬' '(' F ')' | (' F ')'
            elif root[-2].label() == 'F':                

                isnegated_rule = isinstance(root[0], str) and (root[0] == '¬')
                res = cls.from_nltk_tree(root[-2], name2ind_dict, input_vecs)
                
                if isnegated_rule:
                    res = ~res
                    
                return res
            
            # the case F -> F OP F
            elif root[-2].label() == 'OP':
                
                p, q = cls.from_nltk_tree(root[0], name2ind_dict, input_vecs), \
                       cls.from_nltk_tree(root[-1], name2ind_dict, input_vecs)
                
                op = root[1][0]
                if op == '⊕':
                    return np.logical_xor(p, q)
                elif op == '∨':
                    return np.logical_or(p, q)
                elif op == '∧':
                    return np.logical_and(p, q)
                elif op == '→':
                    return np.logical_or(~p, q)
                elif op == '↔':
                    return np.logical_or(np.logical_and(p, q), np.logical_and(~p, ~q))
                else:
                    raise ValueError

        elif label == 'L':

            isnegated_literal = isinstance(root[0], str) and (root[0] == '¬')            

            literal_str = ''.join(root.leaves())
            # remove the possible negation at the beginning
            literal_str = literal_str[1:] if isnegated_literal else literal_str
            
            vec = input_vecs[:, name2ind_dict[literal_str]]
            
            if isnegated_literal:
                vec = ~vec
            
            return vec

        else:
            raise ValueError
    
    @classmethod
    def find_best_LE_score(
            cls,
            true_root,
            pred_root,
            soft_binding: bool,
            greedy_match: bool,
            top_n: int,
            verbose: bool = False
    ):
        """
            Given the groundtruth and the predicted nltk FOL trees, compute the truth tables over all 
            literal bindings and returns the best one
        """

        # first we find "inputs" in each tree, i.e. the set of unique literals in a FOL
        true_inputs, pred_inputs = set(), set()
        VecRuleEvaluator.find_inputs(true_root, true_inputs), VecRuleEvaluator.find_inputs(pred_root, pred_inputs)
        true_inputs, pred_inputs = list(true_inputs), list(pred_inputs)
        n_true_inputs, n_pred_inputs = len(true_inputs), len(pred_inputs)
        min_n, max_n = sorted([n_true_inputs, n_pred_inputs])
        
        best_score, best_binded_pred_inputs = 0., None

        # once we found the inputs, then we deal with the case where # inputs in two trees are different
        # either we do soft binding by adding dummy inputs to the shorter ones, or we simply return 0
        if n_true_inputs != n_pred_inputs:
            if soft_binding:
                # extend the shorter inputs to the max number of inputs by adding dummy input names
                ls_to_extend = true_inputs if n_true_inputs < max_n else pred_inputs
                ls_to_extend.extend([f'{cls.dummy_input_str}_{ind}' for ind in range(max_n-min_n)])
            else:
                return best_score, true_inputs, best_binded_pred_inputs

        # at this point, we have two list ofs inputs of the same length and we will find the input binding that yields
        # the best score
        input_vecs = VecRuleEvaluator.gen_input_vecs(len(true_inputs))
        true_name2ind_dict = dict((e, ind) for ind,e in enumerate(true_inputs))
        true_res_vec = VecRuleEvaluator.from_nltk_tree(true_root, true_name2ind_dict, input_vecs)

        ind_binding_enumerator = \
            cls.enumerate_bindings_with_greedy_match(true_inputs, pred_inputs, top_n) if greedy_match \
            else permutations(list(range(max_n)))

        for cnt_ind, binded_pred_inputs_inds in enumerate(ind_binding_enumerator):
            binded_pred_inputs = [pred_inputs[ind] for ind in binded_pred_inputs_inds]
            pred_name2ind_dict = dict((e, ind) for ind, e in enumerate(binded_pred_inputs))
            pred_res_vec = VecRuleEvaluator.from_nltk_tree(pred_root, pred_name2ind_dict, input_vecs)
            score = (pred_res_vec == true_res_vec).mean(dtype=np.float32).item()

            if verbose:
                print('{0}\n{1}\n{2}\n---\n'.format(
                    score, true_inputs, binded_pred_inputs)
                )

            if score > best_score:
                best_score = score
                best_binded_pred_inputs = binded_pred_inputs

            if cnt_ind + 1 >= top_n:
                break

        return best_score, true_inputs, best_binded_pred_inputs
