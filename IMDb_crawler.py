# Usage: IMDb-crawler.py [-h] --id ID [--threads THREADS] [--test_mode]
# -h, --help                                                        : show this help message and exit
# --id ID                                                           : IMDb ID of the movie (required)
# --threads THREADS                                                 : number of threads to use for crawling (optional, default: 4)
# --test_mode                                                       : enable test mode for faster execution while testing (optional, default: False)
# Example usage:
# python IMDb-crawler.py --id tt15398776                            : the default mode with 4 threads and test_mode disabled
# python IMDb-crawler.py --id tt15398776 --test_mode                : test_mode enabled using the default number of threads
# python IMDb-crawler.py --id tt15398776 --threads 8                : number of threads set to 8 and test_mode disabled
# python IMDb-crawler.py --id tt15398776 --test_mode --threads 8    : test_mode enabled using 8 threads
# python IMDb-crawler.py --id tt15398776 tt0092099 --threads 8     : crawling multiple IMDb IDs with 8 threads

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import json
import os
import threading
import argparse

class IMDb_crawler:
    def __init__(self, id, num_of_threads=4, test_mode=False):
        self.url = f"https://www.imdb.com/title/{id}/"
        self.json_file = ""
        self.data = {
            "metadata": {
                "movie_title": "",
                "movie_imdb_id": f"{id}",
                "reviews_count": 0
            },
            "reviews": {}
        }
        self.test_mode = test_mode
        self.num_of_threads = num_of_threads
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless=new")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-popup-blocking")
        self.chrome_options.add_argument("--disable-infobars")
        self.chrome_options.add_argument("--start-maximized")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.maximize_window()
        self.current_loaded = 0
        self.spoiler_button_clicks = 0
        self.see_more_button_clicks = 0
        print("IMDb crawler initialized with the following parameters:")
        print(f"IMDb ID: {id} | Number of threads: {num_of_threads} | Test mode: {test_mode}")

    def open_browser(self):
        self.driver.get(self.url)
        time.sleep(3)
        print(f"opened browser at {self.url}")
    
    def close_browser(self):
        self.driver.quit()
        print("closed browser")

    def change_url(self, id):
        self.url = f"https://www.imdb.com/title/{id}/"
        self.open_browser()

    def change_to_reviews(self):
        if not self.url.endswith("reviews/"):
            self.url = self.url + "reviews/"
        self.open_browser()
    
    def load_json(self):
        processed_title = "-".join(self.data["metadata"]["movie_title"].split())
        processed_title = "".join(c for c in processed_title if c.isalnum() or c == "-")
        processed_title = "-".join(s for s in processed_title.split("-") if len(s) > 0)
        self.json_file = f"data/{processed_title}.json"
        print(f"data will be written to json file: {self.json_file}")
        if os.path.exists(self.json_file) and os.path.getsize(self.json_file) > 0:
            with open(self.json_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            cnt = 0
            for rating in self.data["reviews"]:
                cnt += len(self.data["reviews"][rating])
            print(f"loaded {cnt} reviews from {self.json_file}")

    def write_json(self):
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"data written to {self.json_file}")

    def scroll_to_bottom(self):
        time.sleep(0.5)
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            ActionChains(self.driver).send_keys(Keys.PAGE_DOWN).perform()
            time.sleep(0.5)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        time.sleep(0.5)

    def get_metadata(self):
        movie_title = self.driver.find_element(By.XPATH, '//*[@id="__next"]/main/div/section/section/div[3]/section/section/div[2]/hgroup/h2').text.strip()
        reviews_count = int(self.driver.find_element(By.XPATH, '//*[@id="__next"]/main/div/section/div/section/div/div[1]/section[1]/div[1]/div/div').text.split()[0].replace(",", ""))
        self.data["metadata"]["movie_title"] = movie_title
        self.data["metadata"]["reviews_count"] = reviews_count
        print(f"movie title: {movie_title} | reviews count: {reviews_count}")

    def get_reviews(self):
        total_reviews = self.data["metadata"]["reviews_count"]
        print(f"number of reviews: {total_reviews}")
        
        self.scroll_to_bottom()

        if self.test_mode:
            total_reviews = 100

        while self.current_loaded < total_reviews:
            self.current_loaded += 25
            self.current_loaded = min(self.current_loaded, total_reviews)
            print(f"loaded {self.current_loaded} out of {total_reviews} reviews")

            try:
                spoiler_buttons = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "review-spoiler-button"))
                )
                print(f"found {len(spoiler_buttons)} spoiler buttons")

                for button in spoiler_buttons:
                    try:
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(0.5)  # give the content some time to expand
                        print(f"clicked spoiler button #{self.spoiler_button_clicks+1}")
                        self.spoiler_button_clicks += 1
                    except Exception as e:
                        print(f"error while clicking spoiler button: {e}")
            except TimeoutException:
                print("review-spoiler-button not found or loading timed out")

            try:
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/main/div/section/div/section/div/div[1]/section[1]/div[3]/div/span[1]/button'))
                )
                try:
                    self.driver.execute_script("arguments[0].click();", element)
                    time.sleep(0.5)
                    print(f"clicked see more button #{self.see_more_button_clicks+1}")
                    self.see_more_button_clicks += 1
                except Exception as e:
                    print(f"error while clicking see more button: {e}")
            except TimeoutException:
                print("see more button not found or loading timed out")

            self.scroll_to_bottom()

        try:
            reviews = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "user-review-item"))
            )
            print(f"found {len(reviews)} reviews")

            lock = threading.Lock()

            def parse_reviews(reviews, num):
                for idx, review in enumerate(reviews):
                    try:
                        rating = review.find_element(By.XPATH, './/span[contains(@class, "ipc-rating-star--rating")]').text
                        title = review.find_element(By.XPATH, './/h3[contains(@class, "ipc-title__text")]').text.strip()
                        content = review.find_element(By.XPATH, './/div[contains(@class, "ipc-html-content-inner-div")]').text.strip()
                        with lock:
                            if self.data["reviews"].get(rating) is None:
                                self.data["reviews"].update({rating: []})
                            self.data["reviews"][rating].append({
                                "title": title,
                                "content": content
                            })
                        print(f"Thread #{num}: review #{idx + 1} crawled")
                    except NoSuchElementException:
                        print(f"Thread #{num}: review #{idx + 1} rating not found, skipped")

            chunks = [reviews[i::self.num_of_threads] for i in range(self.num_of_threads)]
            threads = []
            for i, chunk in enumerate(chunks):
                thread = threading.Thread(target=parse_reviews, args=(chunk,i,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        except TimeoutException:
            print("reviews not found or loading timed out")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDb Crawler")
    parser.add_argument("--id", type=str, nargs='+', required=True, help="IMDb ID(s) of the movie(s)")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use for crawling (default: 4)")
    parser.add_argument("--test_mode", action='store_true', help="Enable test mode for faster execution")
    args = parser.parse_args()

    start_time = time.time()

    for imdb_id in args.id:
        print(f"Starting crawl for IMDb ID: {imdb_id}")
        # Create a new instance of the IMDb_crawler for each ID
        crawler = IMDb_crawler(imdb_id, args.threads, args.test_mode)
        # crawler.open_browser()
        crawler.change_to_reviews()
        crawler.get_metadata()
        crawler.load_json()
        crawler.get_reviews()
        crawler.write_json()
        crawler.close_browser()
        print(f"Finished crawl for IMDb ID: {imdb_id}\n")

    end_time = time.time()
    print(f"started at {time.ctime(start_time)}")
    print(f"finished at {time.ctime(end_time)}")
    print(f"total time: {end_time - start_time} seconds")

