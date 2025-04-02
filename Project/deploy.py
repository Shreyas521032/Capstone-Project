import os
import subprocess
import webbrowser
from pathlib import Path

def check_git_installed():
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def initialize_git_repo():
    print("🔄 Initializing Git repository...")
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit for Streamlit deployment"], check=True)

def create_requirements():
    requirements = """
streamlit>=1.22.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.11.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
statsmodels>=0.13.0
openpyxl>=3.0.0
    """
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    print("✅ Created requirements.txt")

def create_readme():
    readme = 
    with open("README.md", "w") as f:
        f.write(readme)

def setup_github(repo_name):
    print(f"🚀 Creating GitHub repository: {repo_name}")
    subprocess.run(["gh", "repo", "create", repo_name, "--public"], check=True)
    subprocess.run(["git", "remote", "add", "origin", f"git@github.com:yourusername/{repo_name}.git"], check=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
    return f"https://github.com/yourusername/{repo_name}"

def deploy_streamlit(github_url):
    streamlit_url = "https://share.streamlit.io/deploy"
    print(f"🌐 Opening Streamlit deployment page: {streamlit_url}")
    webbrowser.open_new_tab(f"{streamlit_url}?repository={github_url}")

def main():
    print("\n" + "="*50)
    print("🚀 Streamlit App Deployment Setup")
    print("="*50 + "\n")

    # 1. Verify Git
    if not check_git_installed():
        print("❌ Git is not installed. Please install Git first.")
        return

    # 2. Create necessary files
    if not Path("requirements.txt").exists():
        create_requirements()
    
    if not Path("README.md").exists():
        create_readme()

    # 3. Initialize Git if needed
    if not Path(".git").exists():
        initialize_git_repo()
    else:
        print("✅ Existing Git repository found")

    # 4. Get repo name
    repo_name = input("Enter GitHub repository name (e.g., healthcare-analytics): ").strip()
    
    # 5. Setup GitHub
    try:
        github_url = setup_github(repo_name)
        print(f"✅ GitHub repository created: {github_url}")
    except Exception as e:
        print(f"❌ GitHub setup failed: {str(e)}")
        return

    # 6. Deploy to Streamlit
    deploy_streamlit(github_url)
    print("\n✔️ Deployment setup complete!")
    print("1. Complete the deployment form in your browser")
    print("2. Set the main file path to your Streamlit app (e.g., app.py)")
    print("3. Click 'Deploy'\n")

if __name__ == "__main__":
    main()
