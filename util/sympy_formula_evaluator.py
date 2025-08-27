import sympy
from sympy import Expr, Symbol, Function

import random
from typing import Dict
from tqdm.auto import tqdm

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class SymPyFormulaEvaluator:
    def __init__(
            self,
            num_of_random_check: int = 100,
    ):
        # 乱数を代入して等価性を評価する際に試行する回数
        self._num_of_random_check = num_of_random_check

    @staticmethod
    def assign_random_values(symbols):
        values: Dict[sympy.Symbol, float] = {}
        for symbol in symbols:
            values[symbol] = random.uniform(-10, 10)
        return values

    @staticmethod
    def sympify_formula(formula_str: str):
        try:
            return sympy.sympify(formula_str)
        except (sympy.SympifyError, TypeError, AttributeError):
            return None

    @staticmethod
    def simplify_and_expand(formula):
        try:
            return sympy.expand(sympy.simplify(formula))
        except (sympy.SympifyError, AttributeError, TypeError):
            return None

    @staticmethod
    def is_valid_sympy_formula(expr):
        """SymPyの数式かどうかを判定"""
        return isinstance(expr, Expr) and not isinstance(expr, (bool, int, float))

    def random_evaluation(self, formula1, formula2):
        """
        ランダムに値を代入して、1式、2式がそれぞれ0ではなく、かつ差分式が0に近いかを判定する。
        複数回試行するのはランダムな数値によっては以下のようなケースがあるから。

        case1:
            rの値が比率を示している(0~1)とき、1以上の値を代入するとans_1とans_2で符号が入れ替わる例
            ans_1 = "2*(pi*(1/(-omega**2 + (4/3)*(pi*(G*rho*(1 - 2*(r/R)**3)))))**(1/2))"
            ans_2 = "(2*pi)/(sqrt(-omega**2 + ((4*(pi*(G*rho)))/3)*(1 - 2*r**3/(R**3))))"

        Parameters
        ----------
        formula1
        formula2

        Returns
        -------
            条件を満たす代入があればTrue、なければFalse
        """
        try:
            diff_formula = formula1 - formula2
            symbols = diff_formula.free_symbols

            for i in range(self._num_of_random_check):
                values = self.assign_random_values(symbols)
                val_diff = diff_formula.evalf(subs=values)
                val1 = formula1.evalf(subs=values)
                val2 = formula2.evalf(subs=values)

                is_diff_zero = abs(val_diff) < 1e-6
                is_val1_nonzero = abs(val1) > 1e-6
                is_val2_nonzero = abs(val2) > 1e-6
                if is_diff_zero and is_val1_nonzero and is_val2_nonzero:
                    return True
            return False

        except (sympy.SympifyError, AttributeError, TypeError):
            return None

    def evaluate(
            self,
            formula_str1: str,
            formula_str2: str,
    ) -> bool:

        # 文字数が長すぎるものを排除（300文字）
        # 文字数が長すぎるものはsympyでの検証が困難である上に、そもそも問題として不適切である可能性があるので排除する。
        length_threshold = 300
        if len(formula_str1)> length_threshold:
            logger.debug("formula_str1の文字数が長すぎます。等価性を否定します。")
            return False

        if len(formula_str2)> length_threshold:
            logger.debug("formula_str2の文字数が長すぎます。等価性を否定します。")
            return False

        formula1 = self.sympify_formula(formula_str1)
        if formula1 is None:
            logger.debug("formula_str1のsympy変換に失敗しました。等価性を否定します。")
            return False

        formula2 = self.sympify_formula(formula_str2)
        if formula2 is None:
            logger.debug("formula_str2のsympy変換に失敗しました。等価性を否定します。")
            return False

        formula1 = self.simplify_and_expand(formula1)
        formula2 = self.simplify_and_expand(formula2)

        ''' 演算可能な数式となっているか '''
        if not self.is_valid_sympy_formula(formula1):
            logger.debug("formula1のsympy変換に失敗しました。等価性を否定します。")
            return False

        if not self.is_valid_sympy_formula(formula2):
            logger.debug("formula2のsympy変換に失敗しました。等価性を否定します。")
            return False
        ''' 2式のシンボルを確認して一致していなければその時点で不整合 '''
        if not formula1.free_symbols == formula2.free_symbols:
            logger.debug("式のシンボルが一致しませんでした。等価性を否定します。")
            return False

        ''' 2式の関数定義を確認して一致していなければその時点で不整合 '''
        if not formula1.atoms(Function) == formula2.atoms(Function):
            logger.debug("式の関数定義が一致しませんでした。等価性を否定します。")
            return False
        ''' 1. sympy変換式による等価性の確認 '''
        diff_formula = formula1 - formula2
        simplified_diff = sympy.simplify(diff_formula)
        if simplified_diff == 0:
            logger.debug("式の差が0に簡略化されました。等価と判定します。")
            return True

        ''' 2. ランダム代入による等価性の確認 '''
        if self.random_evaluation(formula1, formula2):
            logger.debug("ランダム代入による評価で一致が認められました。等価と判定します。")
            return True

        logger.debug(
            f"{self._num_of_random_check}回の試行で一致が認められませんでした。数式は一致していないと判定します。")
        return False


# 使用例
if __name__ == "__main__":
    """
    python -m util.sympy_formula_evaluator
    """

    from logging import DEBUG, INFO, WARNING, ERROR, basicConfig

    # basicConfig(level=WARNING)
    # basicConfig(level=INFO)
    basicConfig(level=DEBUG)

    # 2つの数式の差を計算し、その差が0になるかで等価性を判定する例
    ans_1 = "2*(pi*(1/(-omega**2 + (4/3)*(pi*(G*rho*(1 - 2*(r/R)**3)))))**(1/2))"
    ans_2 = "(2*pi)/(sqrt(-omega**2 + ((4*(pi*(G*rho)))/3)*(1 - 2*r**3/(R**3))))"

    # ans_1 = "a**2+2*a*b+b**2"
    # ans_2 = "(a+b)**2"

    # ans_1 = "2"
    # ans_2 = "3.0"

    # ans_1 = "1/2"
    # ans_2 = "0.5"

    evaluator = SymPyFormulaEvaluator()

    print(f"試行 : 等価性評価結果 = {evaluator.evaluate(ans_1, ans_2)}")
