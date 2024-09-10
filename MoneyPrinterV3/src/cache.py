import os
import json
from typing import List
from config import ROOT_DIR



def get_cache_path() -> str:
    """
    Gets the path to the cache file.

    Returns:
        path (str): The path to the cache folder
    """
    return os.path.join(ROOT_DIR, '.mp')


def get_youtube_cache_path() -> str:
    """
    Gets the path to the YouTube cache file.

    Returns:
        path (str): The path to the YouTube cache folder
    """
    return os.path.join(get_cache_path(), 'youtube.json')

def get_accounts(provider: str) -> List[dict]:
    """
    Gets the accounts from the cache.

    Args:
        provider (str): The provider to get the accounts for

    Returns:
        account (List[dict]): The accounts
    """
    cache_path = ""

    if provider == "twitter":
        cache_path = get_twitter_cache_path()
    elif provider == "youtube":
        cache_path = get_youtube_cache_path()

    if not os.path.exists(cache_path):
        # Create the cache file
        with open(cache_path, 'w') as file:
            json.dump({
                "accounts": []
            }, file, indent=4)

    with open(cache_path, 'r') as file:
        parsed = json.load(file)

        if parsed is None:
            return []
        
        if 'accounts' not in parsed:
            return []

        # Get accounts dictionary
        return parsed['accounts']

def add_account(provider: str, account: dict) -> None:
    """
    Adds an account to the cache.

    Args:
        account (dict): The account to add

    Returns:
        None
    """
    if provider == "twitter":
        # Get the current accounts
        accounts = get_accounts("twitter")

        # Add the new account
        accounts.append(account)

        # Write the new accounts to the cache
        with open(get_twitter_cache_path(), 'w') as file:
            json.dump({
                "accounts": accounts
            }, file, indent=4)
    elif provider == "youtube":
        # Get the current accounts
        accounts = get_accounts("youtube")

        # Add the new account
        accounts.append(account)

        # Write the new accounts to the cache
        with open(get_youtube_cache_path(), 'w') as file:
            json.dump({
                "accounts": accounts
            }, file, indent=4)

def remove_account(account_id: str) -> None:
    """
    Removes an account from the cache.

    Args:
        account_id (str): The ID of the account to remove

    Returns:
        None
    """
    # Get the current accounts
    accounts = get_accounts()

    # Remove the account
    accounts = [account for account in accounts if account['id'] != account_id]

    # Write the new accounts to the cache
    with open(get_twitter_cache_path(), 'w') as file:
        json.dump({
            "accounts": accounts
        }, file, indent=4)


    
def get_results_cache_path() -> str:
    """
    Gets the path to the results cache file.

    Returns:
        path (str): The path to the results cache folder
    """
    return os.path.join(get_cache_path(), 'scraper_results.csv')
