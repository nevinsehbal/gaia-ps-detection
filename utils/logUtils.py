class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# This is a custom print function that allows easy import and use of different print styles
def printCustom(type, msg):
    if(type == "error"):
        printError(msg)
    elif(type == "warning"):
        printWarning(msg)
    elif(type == "success"):
        printSuccess(msg)
    elif(type == "info"):
        printInfo(msg)
    elif(type == "bold"):
        printBold(msg)
    elif(type == "underline"):
        printUnderline(msg)
    elif(type == "header"):
        printHeader(msg)
    else:
        print(msg)

def printError(msg):
    print(bcolors.FAIL + "[E] " + msg + bcolors.ENDC)

def printWarning(msg):
    print(bcolors.WARNING + "[W] " +msg + bcolors.ENDC)

def printSuccess(msg):
    print(bcolors.OKGREEN + "[S] " +msg + bcolors.ENDC)

def printInfo(msg):
    print(bcolors.OKCYAN + "[I] " +msg + bcolors.ENDC)

def printBold(msg):
    print(bcolors.BOLD + msg + bcolors.ENDC)

def printUnderline(msg):
    print(bcolors.UNDERLINE + msg + bcolors.ENDC)

def printHeader(msg):
    print(bcolors.HEADER + msg + bcolors.ENDC)