import re
from pathlib import Path

def round_file_numbers(input_path, output_path, ndigits=3):
    """
    Lit un fichier texte, arrondit tous les nombres (même en notation scientifique)
    tout en conservant les séparateurs d'origine (espaces / tabulations),
    et écrit le résultat dans output_path.
    """

    number_pattern = re.compile(
        r"""
        (?<![#\w])              # éviter les lignes de commentaires et les mots collés
        [-+]?
        (?:
            \d+\.\d*|\.\d+|\d+  # formats: 12.34 / .34 / 12
        )
        (?:[eE][-+]?\d+)?       # notation scientifique
        """,
        re.VERBOSE
    )

    def repl(match):
        num = float(match.group(0))
        rounded = round(num, ndigits)
        # format propre : supprimer trailing zeros inutiles mais garder un 0 si entier
        text = f"{rounded:.{ndigits}f}".rstrip('0').rstrip('.')
        return text if text else "0"

    with Path(input_path).open("r") as fin, Path(output_path).open("w") as fout:
        for line in fin:
            if line.lstrip().startswith("#"):
                fout.write(line)
                continue
            new_line = number_pattern.sub(repl, line)
            fout.write(new_line)


# Exemple d’utilisation :
round_file_numbers("param_FHMM2_new.txt", "param_FHMM2_round.txt", ndigits=3)
