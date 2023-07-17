from dataclasses import dataclass
from typing import List, Optional
import evaluate
from fol_parser import VecRuleEvaluator, parse_text_FOL_to_tree, msplit
from utils import all_exists


@dataclass
class MetricsOutput:
    FOL_bleu: float
    FOL_LE: float
    FOL_true_inputs: Optional[List[str]]
    FOL_binded_pred_inputs: Optional[List[str]]


@dataclass
class UniversalMetricsOutput(MetricsOutput):
    instruct_bleu: float


class Metrics:
    
    def __init__(self):
        self.bleu = evaluate.load("bleu")
        self.FOL_tokenizer = lambda x: msplit(x)[0]        
    
    def compute_FOL_bleu(self, pred_seq: str, true_seq: str):
        min_len = min(map(lambda x: len(self.FOL_tokenizer(x)), [pred_seq, true_seq]))
        res = self.bleu.compute(predictions=[pred_seq], references=[[true_seq]], 
                                tokenizer=self.FOL_tokenizer, max_order=min(4, min_len))
        return res['bleu']
    
    def compute_instruct_bleu(self, pred_seq: str, true_seq: str):
        min_len = min(map(lambda x: len(x.split()), [pred_seq, true_seq]))
        res = self.bleu.compute(predictions=[pred_seq], references=[[true_seq]], 
                                max_order=min(4, min_len))
        return res['bleu']
    
    def compute_LE(self, pred_text_FOL: str, true_text_FOL: str):
        true_root, pred_root = parse_text_FOL_to_tree(true_text_FOL), parse_text_FOL_to_tree(pred_text_FOL)
        
        # parsing true FOL should never fail
        assert true_root is not None, 'failed parsing true text FOL %s' % true_text_FOL
        
        # parsing pred FOL can fail if model produces invalid rule, in which case, LE score is 0
        if pred_root is None:
            return 0., None, None
        
        # if both parsed successfully, then compute LE score
        score, true_inputs, binded_pred_inputs = \
            VecRuleEvaluator.find_best_LE_score(
                true_root,
                pred_root,
                soft_binding=True,
                greedy_match=True,
                top_n=1000
            )
        return score, true_inputs, binded_pred_inputs
    
    def evaluate(self, pred_seq: str, true_seq: str):
        raise NotImplementedError


class UniversalMetrics(Metrics):

    def evaluate(self, orig_comments: Optional[str], orig_fol: str, pred_comments: str, pred_fol: str):

        FOL_bleu = self.compute_FOL_bleu(pred_fol, orig_fol)
        FOL_eval = self.compute_LE(pred_fol, orig_fol)

        instruct_bleu = None
        # this is a correction metrics
        if all_exists(orig_comments):
            instruct_bleu = self.compute_instruct_bleu(pred_comments, orig_comments)

        return UniversalMetricsOutput(
            instruct_bleu=instruct_bleu,
            FOL_bleu=FOL_bleu,
            FOL_LE=FOL_eval[0],
            FOL_true_inputs=FOL_eval[1],
            FOL_binded_pred_inputs=FOL_eval[2]
        )


