from main import text_to_images
import time

if __name__ == '__main__':
    start = time.time()
    text = "hotel"
    text_to_images(text, save_res=True, count=10)
    time_spent = time.time() - start
    print("Completed api call.Time spent {0:.3f} s".format(time_spent))
