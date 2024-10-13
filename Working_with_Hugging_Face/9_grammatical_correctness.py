from transformers import pipeline

# Create a pipeline
classifier = pipeline(
  task="text-classification", 
  model="abdulmatinomotoso/English_Grammar_Checker"
)

print("---------------------------------------------------------")
print("Ben, kedi gezer :", classifier("Ben, kedi gezer"))
print("Ben, kedi gez. :", classifier("Ben, kedi gez."))
print("I will walk dog :", classifier("I will walk dog"))