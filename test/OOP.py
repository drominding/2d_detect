# 面向对象编程：银行账户操作
 
class BankAccount:
    def __init__(self, account):
        print('__init__ is run')
        self.account = account
        self.balance = 0
 
    def deposit(self, amount):
        print('deposit is run')
        if amount > 0:
            self.balance += amount
            print(f"Deposited {amount}. New balance is {self.balance}.")
        else:
            print("Deposit amount must be positive.")
 
    def withdraw(self, amount):
        print('withdraw is run')
        if 0 < amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew {amount}. New balance is {self.balance}.")
        else:
            print("Insufficient funds or invalid withdrawal amount.")
 
# 使用示例
index = 0
while True:
    action = input('new or deposit or withdraw')
    if action == 'new':
        BankAccount(index)
        index += 1
    
