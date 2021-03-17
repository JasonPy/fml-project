from feature_selection import feature_selection
import time

start = time.time()
print("start")
feature_selection("PCA")
end = time.time()
print(end - start)
print("Finished")