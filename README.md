# Setup Instructions

## Step 1: Clone Repository
```bash
git clone https://github.com/GithubAnant/ARGO.git
cd ARGO
```

## Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 3: Setup API Key
1. Create a new file called `.env` in your project folder
2. Open the `.env` file and add this inside:
```
OPENROUTER_KEY=
```

3. Go to [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) and create your API key
4. Copy your API key and paste it after the `=` sign in your `.env` file:
```
OPENROUTER_KEY=sk-or-v1-your-actual-key-here
```

## Step 4: Run the Application
```bash
python main.py
```

**Note:** First run will take 8-10 minutes to complete as it creates the RAG file.
