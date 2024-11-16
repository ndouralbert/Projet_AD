import os
import subprocess

def run_git_lfs_commands():
    try:
        # Migration du fichier vers Git LFS
        print("Migration du fichier vers Git LFS...")
        subprocess.run(["git", "lfs", "migrate", "import", "--include=src/globalterrorismdb_0718dist.csv"], check=True)
        
        # Ajouter tous les fichiers non suivis
        print("Ajout des fichiers non suivis...")
        subprocess.run(["git", "add", "."], check=True)  # Ajoute tous les fichiers

        # Validation des changements
        print("Validation des changements...")
        subprocess.run(["git", "commit", "-m", "Migrer le fichier CSV vers Git LFS"], check=True)
        
        # Pousser vers le dépôt distant
        print("Pousser les changements vers le dépôt distant...")
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)

        # Lister les fichiers suivis par LFS
        print("Fichiers suivis par LFS :")
        subprocess.run(["git", "lfs", "ls-files"], check=True)

        print("Les commandes Git ont été exécutées avec succès.")
    
    except subprocess.CalledProcessError as e:
        print(f"Une erreur est survenue lors de l'exécution des commandes Git : {e}")

if __name__ == "__main__":
    run_git_lfs_commands()