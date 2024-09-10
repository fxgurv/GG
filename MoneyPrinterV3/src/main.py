import schedule
import subprocess
from art import *
from cache import *
from utils import *
from config import *
from status import *
from uuid import uuid4
from constants import *
from classes.Tts import TTS
from termcolor import colored
from classes.YouTube import YouTube
from prettytable import PrettyTable
from save_short import save_short
from upload_youtube_sele import upload_youtube_using_selenium

def main():

    # Get user input
    valid_input = False
    while not valid_input:
        try:
            # Show user options
            info("\n============ OPTIONS ============", False)

            for idx, option in enumerate(OPTIONS):
                print(colored(f" {idx + 1}. {option}", "cyan"))

            info("=================================\n", False)
            user_input = input("Select an option: ").strip()
            if user_input == '':
                print("\n" * 100)
                raise ValueError("Empty input is not allowed.")
            user_input = int(user_input)
            valid_input = True
        except ValueError as e:
            print("\n" * 100)
            print(f"Invalid input: {e}")

    # Start the selected option
    if user_input == 1:
        info("Starting YT Shorts Automater...")

        cached_accounts = get_accounts("youtube")

        if len(cached_accounts) == 0:
            warning("No accounts found in cache. Create one now?")
            user_input = question("Yes/No: ")

            if user_input.lower() == "yes":
                generated_uuid = str(uuid4())

                success(f" => Generated ID: {generated_uuid}")
                nickname = question(" => Enter a nickname for this account: ")
                fp_profile = question(" => Enter the path to the Firefox profile: ")
                niche = question(" => Enter the account niche: ")
                language = question(" => Enter the account language: ")

                add_account("youtube", {
                    "id": generated_uuid,
                    "nickname": nickname,
                    "firefox_profile": fp_profile,
                    "niche": niche,
                    "language": language,
                    "videos": []
                })
        else:
            table = PrettyTable()
            table.field_names = ["ID", "UUID", "Nickname", "Niche"]

            for account in cached_accounts:
                table.add_row([cached_accounts.index(account) + 1, colored(account["id"], "cyan"), colored(account["nickname"], "blue"), colored(account["niche"], "green")])

            print(table)

            user_input = question("Select an account to start: ")

            selected_account = None

            for account in cached_accounts:
                if str(cached_accounts.index(account) + 1) == user_input:
                    selected_account = account

            if selected_account is None:
                error("Invalid account selected. Please try again.", "red")
                main()
            else:
                niche = get_niche()
                while niche:
                    print('niche: ', niche)
                    youtube = YouTube(
                        selected_account["id"],
                        selected_account["nickname"],
                        selected_account["firefox_profile"],
                        niche,
                        selected_account["language"]
                    )
                    rem_temp_files()
                    tts = TTS()
                    youtube.generate_video(tts)
                    save_short(youtube.video_path, youtube.metadata['title'], youtube.metadata['description'])
                    niche = get_niche()
                return
    else:
        error("Invalid option selected. Please try again.", "red")
        main()

if __name__ == "__main__":
    # Print ASCII Banner
    print_banner()

    first_time = get_first_time_running()

    if first_time:
        print(colored("Hey! It looks like you're running MoneyPrinter V2 for the first time. Let's get you setup first!", "yellow"))

    # Setup file tree
    assert_folder_structure()

    # Remove temporary files
    rem_temp_files()

    # Fetch MP3 Files
    fetch_songs()

    main()
