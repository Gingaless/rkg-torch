from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import random

def login_for_cookies(config):
    login_url='https://accounts.pixiv.net/login'
    #pixiv_root="https://www.pixiv.net/"
    #service_args=[]

    driver = webdriver.Chrome()
    driver.get(login_url)

    try:
        element = WebDriverWait(driver, 300).until(
            EC.presence_of_element_located((By.ID, "root"))
            )
    except Exception as e:
        driver.save_screenshot('./temp/login_err.png')
        driver.quit()
        raise IOError("login sim wait failed, 'root' did not appear")

    cookies_dict=dict()
    cookies=driver.get_cookies()
    for cookie in cookies:
        cookies_dict[cookie['name']] = cookie['value']

    webdriver.ActionChains(driver).pause(random.uniform(0.5,1.5)).perform()
    driver.quit()
        
    return cookies_dict