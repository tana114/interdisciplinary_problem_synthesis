from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, TypeVar, Generic
import json
from pathlib import Path
from pydantic import BaseModel, Field

# from client.concrete.prompt_loader import load_prompt_config
from client.client_base import ApiClientBase
from util.file_tools_gen import PromptConfigAnalyzer

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())

PROMPTS_DIR = Path(__file__).parent / "prompts"
prompt_analyzer = PromptConfigAnalyzer(PROMPTS_DIR)
prompt_config = prompt_analyzer.load_config(version='1.0', prompt_type="math_phys_v1")

SYSTEM_PROMPT_FORMAT = prompt_config['prompts']['system']
USER_PROMPT_FORMAT = prompt_config['prompts']['user_template']

# prompt_config = load_prompt_config(version='1.0', prompt_type="math_phys_v1")
# SYSTEM_PROMPT_FORMAT = prompt_config['prompts']['system']
# USER_PROMPT_FORMAT = prompt_config['prompts']['user_template']

''' SYSTEM_PROMPT_FORMAT
{sample_num}は、few-shotとして渡すサンプルの数
'''


class MathProblemData(BaseModel):
    """Synthesised math problem"""
    Problem_Draft: str = Field(description="The problem synthesised with reference to two skills.")
    Elements_Identified: str = Field(description="result of Step 1")
    Plan: str = Field(description="result of Step 2")
    Rewritten_Problem: str = Field(description="result of Step 3")
    Rewritten_Score: float = Field(description="Rewritten_Problem evaluation value")
    Final_Rewritten_Problem: str = Field(description="result of Step 5. final rewritten problem.")
    Final_Rewritten_Score: float = Field(description="Final_Rewritten_Problem evaluation value")


class MathPhysProblemGenerator(ApiClientBase):

    def __init__(
            self,
            model_name: str,
            few_shot_num: int,
            api_key: Optional[str] = None,
    ):
        super().__init__(model_name, api_key)
        self._few_shot_num = few_shot_num

    # @staticmethod
    @classmethod
    def encode_few_shot_prompt(
            cls,
            skill: str,
            seed_instructions: List[Dict[str, str]]
    ) -> str:
        """
        Few-Shotプロンプト用の文字を生成する

        skill = "algebra"
        seed_instructions =[
            {"problem": "hoge1", "solution": "fuga1"},
            {"problem": "hoge2", "solution": "fuga2"},
            {"problem": "hoge3", "solution": "fuga3"},
        ]

        few_shot_prompts = '''
        algebra
            {
                "no": 1,
                "skill": "algebra",
                "problem": "hoge1",
                "solution": "fuga1"
            }
            ...
            {
                "no": 3,
                "skill": "algebra",
                "problem": "hoge3",
                "solution": "fuga3"
            }
        '''

        """
        seeds = [{"skill": skill, **d} for d in seed_instructions]
        seeds = [{"no": i + 1, **d} for i, d in enumerate(seeds)]

        few_shot_prompt = f"{skill}\n"
        # convert dict type to string.
        for d in seeds:
            few_shot_prompt += json.dumps(d, indent=2, ensure_ascii=False)
            few_shot_prompt += "\n"

        return few_shot_prompt

    def _create_message_config(self, prompt_elements: Dict):
        pass

    def _create_parse_config(self, prompt_elements: Dict) -> Dict:
        system_prompt = SYSTEM_PROMPT_FORMAT.format(sample_num=self._few_shot_num)
        user_prompt = USER_PROMPT_FORMAT.format(sample_num=self._few_shot_num)

        # prompt_elements -> Dict[Literal["few_shot_1", "few_shot_2"], str],
        user_prompt = user_prompt.format(**prompt_elements)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        config = {
            "model": self._model_name,
            "messages": messages,
            "response_format": MathProblemData,
        }

        return config


if __name__ == "__main__":
    """
    python -m client.concrete.math_phys_problem_gen
    """

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    # basicConfig(level=WARN)
    basicConfig(level=INFO)

    skill_1 = 'algebra'
    seeds_1 = [
        {
            "problem": r"Let $\lfloor x\rfloor$ represent the greatest integer which is less than or equal to $x$. For example, $\lfloor 3\rfloor=3,\lfloor 2.6\rfloor=2$. If $x$ is positive and $x\lfloor x\rfloor=17$, what is the value of $x$ ?",
            "solution": "We deduce that $4<x<5$.\n\nOtherwise, if $x \\leq 4, x\\lfloor x\\rfloor \\leq 16$, and if $x \\geq 5, x\\lfloor x\\rfloor \\geq 25$.\n\nTherefore $\\lfloor x\\rfloor=4$\n\nSince $x\\lfloor x\\rfloor=17$\n\n$$\n\\begin{aligned}\n4 x & =17 \\\\\nx & =4.25\n\\end{aligned}\n$$",
        },
        {
            "problem": r"If $2 \leq x \leq 5$ and $10 \leq y \leq 20$, what is the maximum value of $15-\frac{y}{x}$ ?",
            "solution": (
                "Since we want to make $15-\\frac{y}{x}$ as large as possible, then we want to subtract as little as possible from 15.\n\nIn other words, we want to make $\\frac{y}{x}$ as small as possible.\n\nTo make a fraction with positive numerator and denominator as small as possible, we make the numerator as small as possible and the denominator as large as possible.\n\nSince $2 \\leq x \\leq 5$ and $10 \\leq y \\leq 20$, then we make $x=5$ and $y=10$.\n\nTherefore, the maximum value of $15-\\frac{y}{x}$ is $15-\\frac{10}{5}=13$."
                "Since $y$ is positive and $2 \\leq x \\leq 5$, then $15-\\frac{y}{x} \\leq 15-\\frac{y}{5}$ for any $x$ with $2 \\leq x \\leq 5$ and positive $y$.\n\nSince $10 \\leq y \\leq 20$, then $15-\\frac{y}{5} \\leq 15-\\frac{10}{5}$ for any $y$ with $10 \\leq y \\leq 20$.\n\nTherefore, for any $x$ and $y$ in these ranges, $15-\\frac{y}{x} \\leq 15-\\frac{10}{5}=13$, and so the maximum possible value is 13 (which occurs when $x=5$ and $y=10$ )."),
        },
        {
            "problem": r"The equations $y=a(x-2)(x+4)$ and $y=2(x-h)^{2}+k$ represent the same parabola. What are the values of $a, h$ and $k$ ?",
            "solution": (
                "We expand the right sides of the two equations, collecting like terms in each case:\n\n$$\n\\begin{aligned}\n& y=a(x-2)(x+4)=a\\left(x^{2}+2 x-8\\right)=a x^{2}+2 a x-8 a \\\\\n& y=2(x-h)^{2}+k=2\\left(x^{2}-2 h x+h^{2}\\right)+k=2 x^{2}-4 h x+\\left(2 h^{2}+k\\right)\n\\end{aligned}\n$$\n\nSince these two equations represent the same parabola, then the corresponding coefficients must be equal. That is, $a=2$ and $2 a=-4 h$ and $-8 a=2 h^{2}+k$.\n\nSince $a=2$ and $2 a=-4 h$, then $4=-4 h$ and so $h=-1$.\n\nSince $-8 a=2 h^{2}+k$ and $a=2$ and $h=-1$, then $-16=2+k$ and so $k=-18$.\n\nThus, $a=2, h=-1$, and $k=-18$."
                "From the equation $y=a(x-2)(x+4)$, we can find the axis of symmetry by calculating the midpoint of the $x$-intercepts.\n\nSince the $x$-intercepts are 2 and -4 , the axis of symmetry is at $x=\\frac{1}{2}(2+(-4))=-1$.\n\nSince the vertex of the parabola lies on the axis of symmetry, then the $x$-coordinate of the vertex is -1 .\n\nTo find the $y$-coordinate of the vertex, we substitute $x=-1$ back into the equation $y=a(x-2)(x+4)$ to obtain $y=a(-1-2)(-1+4)=-9 a$.\n\nThus, the vertex of the parabola is $(-1,-9 a)$.\n\nSince the second equation for the same parabola is in vertex form, $y=2(x-h)^{2}+k$, we can see that the vertex is at $(h, k)$ and $a=2$.\n\nSince $a=2$, the vertex has coordinates $(-1,-18)$, which means that $h=-1$ and $k=-18$. Thus, $a=2, h=-1$ and $k=-18$."),
        },
    ]

    skill_2 = 'atmic'
    seeds_2 = [
        {
            "problem": "(a) The ground state of the hydrogen atom is split by the hyperfine interaction. Indicate the level diagram and show from first principles which state lies higher in energy.\n\n(b) The ground state of the hydrogen molecule is split into total nuclear spin triplet and singlet states. Show from first principles which state lies higher in energy.",
            "solution": "(a) The hyperfine interaction in hydrogen arises from the magnetic interaction between the intrinsic magnetic moments of the proton and the electron, the Hamiltonian being\n\n$$H_{\\text{int}} = -\\mu_p \\cdot \\mathbf{B},$$\n\nwhere $\\mathbf{B}$ is the magnetic field produced by the magnetic moment of the electron and $\\mu_p$ is the intrinsic magnetic moment of the proton.\n\nIn the ground state, the electron charge density is spherically symmetric so that $\\mathbf{B}$ has the same direction as the electron intrinsic magnetic moment $\\mu_e$. However as the electron is negatively charged, $\\mu_e$ is antiparallel to the electron spin angular momentum $s_e$. For the lowest energy state of $H_{\\text{int}}$, $\\langle \\mu_p \\cdot \\mu_e \\rangle > 0$, and so $\\langle s_p \\cdot s_e \\rangle < 0$. Thus the singlet state $F = 0$ is the ground state, while the triplet $F = 1$ is an excited state (see Fig. 1.12).\n\n(b) As hydrogen molecule consists of two like atoms, each having a proton (spin $\\frac{1}{2}$) as nucleus, the nuclear system must have an antisymmetric state function. Then the nuclear spin singlet state ($S = 0$, antisymmetric) must be associated with a symmetric nuclear rotational state; thus $J = 0, 2, 4, \\ldots$, with the ground state having $J = 0$. For the spin triplet state ($S = 1$, symmetric) the rotational state must have $J = 1, 3, \\ldots$, with the ground state having $J = 1$. As the rotational energy is proportional to $J(J + 1)$, the spin triplet ground state lies higher in energy.",
        },
        {
            "problem": "An ARMLbar is a $7 \times 7$ grid of unit squares with the center unit square removed. A portion of an ARMLbar is a square section of the bar, cut along the gridlines of the original bar. Compute the number of different ways there are to cut a single portion from an ARMLbar.",
            "solution": "(a) A spectral series of a hydrogen-like atom has wave numbers \n\n$$\n\\tilde{\\nu} = Z^2 R \\left( \\frac{1}{n^2} - \\frac{1}{m^2} \\right),\n$$\n\nwhere $Z$ is the nuclear charge, $R$ is the Rydberg constant, and $n, m$ are positive integers with $m > n$. The ionization energy of the ground state of H atom is the limit of the Lyman series ($n = 1$), the wave number being\n\n$$\n\\tilde{\\nu}_0 = \\frac{1}{\\lambda_0} = R.\n$$\n\nFor the alpha line of the Lyman series,\n\n$$\n\\tilde{\\nu}_\\alpha = \\frac{1}{\\lambda_\\alpha} = R \\left( 1 - \\frac{1}{2^2} \\right) = \\frac{3}{4}R = \\frac{3}{4\\lambda_0}.\n$$\n\nAs $\\lambda_\\alpha = 1215 \\, \\text{Å}, \\, \\lambda_0 = 3\\lambda_\\alpha/4 = 911 \\, \\text{Å}.$ Hence the wavelength of light that can photoionize H atom in the ground state must be shorter than $911 \\text{Å}$.\n\n(b) The wavelength should be shorter than the limit of the Balmer series $(n = 2)$, whose wave number is\n\n$$\n\\tilde{\\nu} = \\frac{1}{\\lambda} = \\frac{R}{2^2} = \\frac{1}{4\\lambda_0}.\n$$\n\nHence the wavelength should be shorter than $4\\lambda_0 = 3645 \\, \\text{Å}.$\n\n(c) The limiting wave number of the Lyman series of $He^+ \\, (Z = 2)$ is\n\n$$\n\\tilde{\\nu} = \\frac{1}{\\lambda} = \\frac{Z^2 R}{1^2} = 4R = \\frac{4}{\\lambda_0}.\n$$\n\nThe wavelength that can photoionize the $He^+$ in the ground state must be shorter than $\\lambda_0/4 = 228 \\, \\text{Å}. $\n\n(d) The wavelength should be shorter than $1/R = \\lambda_0 = 1215 \\, \\text{Å}.$"
        },
        {
            "problem": "(a) Derive the argument for why heavy nuclei are α-radioactive but stable against neutron-emission.\n\n(b) What methods and arguments are used to determine nuclear radii?\n\n(c) What are the properties that identify a system of nucleons in its lowest energy state? Discuss the nonclassical properties.\n\n(d) The fission cross sections of the following uranium ($Z = 92$) isotopes for thermal neutrons are shown in the table below.\n\n| Isotope | σ (barns) |\n|---------|-----------|\n| $^{230}U$ | 20        |\n| $^{231}U$ | 300       |\n| $^{232}U$ | 76        |\n| $^{233}U$ | 530       |\n| $^{234}U$ | 0         |\n| $^{235}U$ | 580       |\n| $^{236}U$ | 0         |\n\nThe fast-neutron fission cross sections of the same isotopes are all of the order of a few barns, and the even-odd periodicity is much less pronounced. Explain these facts.",
            "solution": "(a) The reason why heavy nuclei only are $\\alpha$-radioactive has been discussed in *Problems 2033 and 2034*. For ordinary nuclei near the $\\beta$-stability curve, the binding energy of the last neutron is positive so that no neutron-radioactivity exists naturally. However, for neutron-rich isotopes far from the $\\beta$-stability curve, the binding energy may be negative for the last neutron, and so neutron-emission may occur spontaneously. As there is no Coulomb barrier for neutrons, emission is a transient process. Also, certain excited states arising from $\\beta$-decays may emit neutrons. In such cases, as the neutron-emission follows a $\\beta$-decay the emitted neutrons are called delayed neutrons. The half-life against delayed-neutron emission is the same as that against the related $\\beta$-decay.\n\n(b) There are two categories of methods for measuring nuclear radii. The first category makes use of the range of the strong interaction of nuclear forces by studying the scattering by nuclei of neutrons, protons or $\\alpha$-particles, particularly by measuring the total cross-section of intermediate-energy neutrons. Such methods give the nuclear radius as\n\n$$\nR = R_0 A^{1/3}, \\quad R_0 \\approx (1.4 \\sim 1.5) \\, \\text{fm}.\n$$\n\nThe other category of methods makes use of the Coulomb interaction between charged particles and atomic nuclei or that among particles within a nucleus to get the electromagnetic nuclear radius. By studying the scattering between high energy electrons and atomic nuclei, the form factors of the nuclei may be deduced which gives the electromagnetic nuclear radius. Assuming mirror nuclei to be of the same structure, their mass difference is caused by Coulomb energy difference and the mass difference between neutron and proton. We have (*Problem 2010*)\n\n$$\n\\Delta E = \\frac{3}{5} \\frac{e^2}{R} (2Z - 1) - (m_n - m_p)c^2\n$$\n\nfor the energy difference between the ground states of the mirror nuclei, which then gives the electromagnetic nuclear radius $R$. A more precise\n\nmethod is to study the deviation of $\\mu$-mesic atom from Bohr's model of hydrogen atom. Because the Bohr radius of the mesic atom is much smaller than that of the hydrogen atom, the former is more sensitive to the value of the electromagnetic nuclear radius, which, by this method, is\n\n$$\nR = R_0 A^{1/3}, \\quad R_0 \\approx 1.1 \\, \\text{fm}.\n$$\n\nHigh-energy electron scattering experiments show that charge distribution within a nucleus is nonuniform.\n\n(c) The ground state of a system of nucleons is identified by its spin, parity and isospin quantum numbers.\n\nSpin and parity are determined by those of the last one or two unpaired nucleons. For the ground state of an even-even nucleus, $J^p = 0^+$. For an even-odd nucleus, the nuclear spin and parity are determined by the last nucleon, and for an odd-odd nucleus, by the spin-orbit coupling of the last two nucleons.\n\nThe isospin of the nuclear ground state is $I = \\frac{1}{2}|N - Z|$.\n\n(d) There is a fission barrier of about 6 MeV for uranium so that spontaneous fission is unlikely and external inducement is required. At the same time, there is a tendency for neutrons in a nucleus to pair up so that isotopes with even numbers of neutrons, $N$, have higher binding energies. When an uranium isotope with an odd number of neutrons captures a neutron and becomes an isotope of even $N$, the excitation energy of the compound nucleus is large, sufficient to overcome the fission barrier, and fission occurs. On the other hand, when an even-$N$ uranium isotope captures a neutron to become an isotope of odd $N$, the excitation energy of the compound nucleus is small, not sufficient to overcome the fission barrier, and fission does not take place. For example, in $^{235}U + n \\rightarrow {}^{236}U^*$ the excitation energy of the compound nucleus $^{236}U^*$ is 6.4 MeV, higher than the fission barrier of ${236}U$ of 5.9 MeV, so the probability of this reaction results in a fission is large. In $^{238}U + n \\rightarrow {}^{239}U^*$, the excitation energy is only 4.8 MeV, lower than the fission barrier of 6.2 MeV of $^{239}U$, and so the probability for fission is low. Such nuclides require neutrons of higher energies to achieve fission. When the neutron energy is higher than a certain threshold, fission cross section becomes large and fission may occur.\n\nThermal neutrons, which can cause fission when captured by odd-$N$ uranium isotopes, have long wavelengths and hence large capture cross sections. Thus the cross sections for fission induced by thermal neutrons are large, in hundreds of barns, for uranium isotopes of odd $N$. They are small for isotope of even $N$.\n\nIf a fast neutron is captured by an uranium isotope the excitation energy of the compound nucleus is larger than the fission barrier and fission occurs irrespective of whether the isotope has an even or an odd number of neu-trons. While fast neutrons have smaller probability of being captured their fission cross section, which is of the order of a few barns, do not change with the even-odd periodicity of the neutron number of the uranium isotope.",
        },
    ]

    gen = MathPhysProblemGenerator(
        # model_name="deepseek/deepseek-r1:free",
        # model_name="deepseek/deepseek-r1-0528:free",
        # model_name="deepseek/deepseek-chat-v3-0324:free",
        model_name="deepseek/deepseek-r1-0528",
        # model_name="deepseek/deepseek-chat-v3-0324",
        few_shot_num=3,
    )

    few_shot_1 = gen.encode_few_shot_prompt(skill_1, seeds_1)
    # print(few_shot_1)

    few_shot_2 = gen.encode_few_shot_prompt(skill_2, seeds_2)
    # print(few_shot_2)

    inst = {
        "few_shot_1": few_shot_1,
        "few_shot_2": few_shot_2,
    }

    res = gen.parse(
        inst,
        temperature=0.7,
        # top_p=0.95,
    )

    print(type(res))
    print(res)

    if res:
        print("math_problem: ", res['Final_Rewritten_Problem'])
