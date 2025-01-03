# 面向对象编程：银行账户操作
 
class BankAccount:
    def __init__(self, id):
        print('__init__ is run')
        self.id = id
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
    def checkid(self):
        print(self.id)
    def show(self):
        print(self.id)
        print(self.balance)
 
# 使用示例
bank = []
id = 0

while True:
    action = input('new or deposit or withdraw\n')
    if action == 'n':
        bank.append(BankAccount(id))
        id += 1
    if action == 'd':
        
        try:
            acccount_id = int(input('input id\n'))
            user = bank[acccount_id]
        except:
            print('wrong id')
            continue
        while True:
            num = input('input a number\n')
            try:
                num = int(num)
                user.deposit(num)
                break
            except:
                print('try again')
        user.show()
    if action == 'check':
        print(bank)
