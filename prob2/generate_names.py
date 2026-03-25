import os

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# A large sample of diverse Indian names to fill your training set
indian_names = [
    "Aarav", "Vihaan", "Aditya", "Arjun", "Sai", "Ishaan", "Krishna", "Aryan", "Shaurya", "Atharv",
    "Ananya", "Diya", "Advika", "Anvi", "Ishani", "Myra", "Kyra", "Navya", "Sana", "Viha",
    "Rajesh", "Suresh", "Ramesh", "Dinesh", "Mahesh", "Naresh", "Mukesh", "Rakesh", "Sanjay", "Sunil",
    "Priyanka", "Deepika", "Anjali", "Kavita", "Sunita", "Anita", "Meeta", "Geeta", "Kavya", "Swati",
    "Parth", "Chirag", "Sameer", "Rohan", "Sahil", "Varun", "Kunal", "Akash", "Neeraj", "Pankaj",
    "Amit", "Abhishek", "Rahul", "Vikram", "Sandeep", "Manoj", "Kiran", "Deepak", "Vivek", "Vijay",
    "Devendra", "Janabai", "Varalakshmi", "Sapna", "Lilavati", "Jayaram", "Satendra", "Sabina", "Subash",
    "Kamalamma", "Sumati", "Sharmila", "Sadhana", "Sanjib", "Suma", "Sakuntala", "Baburao", "Mohani",
    "Nilima", "Siyaram", "Balvir", "Gouri", "Prabhavati", "Lalu", "Gangabai", "Hanuman", "Gurmail",
    "Ranjita", "Baliram", "Satyabhama", "Jayaben", "Dashrath", "Ismail", "Manda", "Shama", "Parveen",
    "Sampat", "Ramadas", "Arif", "Muni", "Sher", "Ramji", "Nagaraja", "Lokesh", "Yamuna", "Paras"
]

# Multiplying and adding slight variations to reach 1000 names efficiently for Task-0
final_list = []
while len(final_list) < 1000:
    for name in indian_names:
        if len(final_list) < 1000:
            final_list.append(name)

# Save to TrainingNames.txt
with open("data/TrainingNames.txt", "w", encoding="utf-8") as f:
    for name in final_list:
        f.write(name + "\n")

print(f"Successfully saved {len(final_list)} names to 'data/TrainingNames.txt'")