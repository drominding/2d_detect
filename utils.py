from colorama import init, Fore, Back, Style

def colorprint(num):
    if num > 0.9:
        color = Fore.RED
    elif num < 0.5:
        color = Fore.GREEN
    else:
        color = Fore.RESET
    print(f'{color}{str(num)}')