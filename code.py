import os

def collect_python_files(directory: str, output_file: str = "code.md"):
    """
    Parcourt un dossier donné, liste les fichiers .py valides,
    et écrit leur contenu dans un fichier Markdown.
    """
    with open(output_file, "w", encoding="utf-8") as md:
        for root, dirs, files in os.walk(directory):
            # Ignorer les dossiers __pycache__
            if "__pycache__" in root:
                continue
            print(f"Collecting {files}")
            for file in files:
                
                # Ignorer __init__.py et fichiers non .py
                if file == "__init__.py":
                    continue
                if not file.endswith(".py"):
                    continue
                if file.endswith(".pyc"):
                    continue

                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    print(f"Impossible de lire {file_path}: {e}")
                    continue

                # Écrire dans le fichier Markdown
                md.write(f"### {file}\n\n")
                md.write("```py\n")
                md.write(content)
                md.write("\n```\n\n")

if __name__ == "__main__":
    # Dossier codé en dur
    dossier = "./soft_position_hmm"  # <-- Remplace par ton chemin
    collect_python_files(dossier)
    print("Fichier code.md généré avec succès.")
