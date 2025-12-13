import os
import dotenv

DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
dotenv.load_dotenv(DOTENV_PATH)