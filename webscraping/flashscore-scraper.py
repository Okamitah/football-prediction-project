# First, let's import our necessary libraries

import time

import pandas as pd

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# And initialise the Selenium Webdriver

service = Service()
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

# I opted to make a function that scrapes a whole season, with a given link, to organise our data how it suits us
# Whether it is to work on multiple leagues in a single season, or a single league during multiple seasons

def scrape_season(season_name, season_link):

    driver.get(season_link)

    # The features we'll gather are listed in this columns list

    columns = ["Home team name", "Away team name", "Outcome", "Home score", "Away score",
               "Home rating", "Away rating", "Home odds", "Draw odds", "Away odds",
               "Home xG", "Away xG", "Home possession", "Away possession",
               "Home lineups", "Away lineups", "Home Formation", "Away formation"
               ]

    # And we'll instantiate our dataframe where we'll organise our data

    matches_data = pd.DataFrame(columns=columns)

    # Sometimes we get a cookie banner in Flashscore

    try:

        cookie_banner = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((
            By.ID, "onetrust-reject-all-handler"))
        )
        cookie_banner.click()

    except:

        print("No cookie banner found or already dismissed.")

    # We'll navigate to the results section of the website, where the season's matches will be listed

    results = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(
        (By.ID, 'li2'))
    )

    # ActionChains is really important in JavaScript heavy websites

    actions = ActionChains(driver)


    try:

        # Some elements can be intercepted by popups, that's why we have to proceed with caution

        actions.move_to_element(results).perform()
        results.click()

    except Exception as e:

        print(":'(")
        print(e)

    time.sleep(2)

    try:

        # We have to extend the visible matches first. In 20 teams leagues, it's 3 clicks

        for i in range(3):

            more_matches = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((
                By.XPATH, "//a[contains(text(),'Show more matches')]")))

            more_matches.click()

            time.sleep(1)

    except:

        print("Already extended")


    time.sleep(2)

    # We'll check how many matches are visible, it depends on each league, and we'll use it to organise out dataframe


    matches = driver.find_elements(By.XPATH, "//div[contains(@id, 'g_1_')]")
    print(f"Matches: {len(matches)}")

    # And now the magic begins: We'll go through the matches, one by one, to extract the necessary data

    for i, match in enumerate(matches[::-1]): # The i will be used for scrolling up with the matches

        try:

            # We'll move to the specified match and click on it, it'll open a new window

            driver.execute_script("arguments[0].scrollIntoView(true);", match)
            driver.execute_script("window.scrollBy(0, -window.innerHeight / 2);")
            actions.move_to_element(match).click().perform()

            # That's why we have to use window_handles, so we can switch to the new window

            windows = driver.window_handles
            driver.switch_to.window(windows[-1]) # Usually the new window will be the last one in the window_handles

            # Now we'll pick our features:

            home_team = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//a[@class='participant__participantName participant__overflow '])[1]"))).text

            away_team = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//a[@class='participant__participantName participant__overflow '])[2]"))).text

            home_score = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='detailScore__wrapper']/span)[1]"))).text

            away_score = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='detailScore__wrapper']/span)[3]"))).text

            home_odds = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='oddsValueInner'])[1]"))).text

            draw_odds = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='oddsValueInner'])[2]"))).text

            away_odds = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='oddsValueInner'])[3]"))).text

            # xG is the expected goals

            home_xG = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-homeValue_-iJBW'])[1]"))).text

            home_possession = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-homeValue_-iJBW'])[2]"))).text

            # SOT is the shots on target

            home_SOT = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-homeValue_-iJBW'])[3]"))).text

            away_xG = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-awayValue_rQvxs'])[1]"))).text

            away_possession = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-awayValue_rQvxs'])[2]"))).text

            away_SOT = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-awayValue_rQvxs'])[3]"))).text

            # Now we'll move to the lineups section of the match, so we get the ratings, formations and lineups:

            lineups_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Lineups')]")

            #Sometimes these buttons are stubborn, so we have to call in the big guns: ActionChains

            actions.move_to_element(lineups_button).perform()
            lineups_button.click()

            # We'll let our driver chill for a bit

            time.sleep(2)

            home_rating = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='wcl-caption_xZPDJ wcl-scores-caption-03_LG4YJ wcl-bold_slHaC'])[1]"))).text

            away_rating = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='wcl-caption_xZPDJ wcl-scores-caption-03_LG4YJ wcl-bold_slHaC'])[2]"))).text

            home_formation = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='lf__headerPart'])[1]"))).text

            away_formation = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='lf__headerPart'])[2]"))).text

            # We'll store the players names elements at first in a list

            lineups_placement = driver.find_elements(By.XPATH,
                    "//span[@class='wcl-caption_xZPDJ wcl-scores-caption-04_tdF7p wcl-captionEllipsis_YrmbS wcl-captionResponsive_lB-gc wcl-bold_slHaC']")

            print(len(lineups_placement))

            # Then we'll start adding the text (player's name) to a list

            home_lineup = []

            for player in lineups_placement[:11]:  # There are 11 players per team !
                home_lineup.append(player.text)

            away_lineup = []

            for player in lineups_placement[11:]:
                away_lineup.append(player.text)

            # For the outcome, we'll make 1 if the home team wins, 2 if the away team wins and 0 if it ends in a draw

            if home_score > away_score:
                outcome = 1
            elif home_score < away_score:
                outcome = 2
            else:
                outcome = 0

            # And now that we've completed the data gathering, we'll add our row to the dataframe:

            row = {"Home team name": home_team, "Away team name": away_team, "Outcome": outcome,
                   "Home score": home_score, "Away score": away_score, "Home rating": home_rating, "Away rating": away_rating,
                   "Home odds": home_odds, "Draw odds": draw_odds, "Away odds": away_odds,
                    "Home xG": home_xG, "Away xG": away_xG, "Home possession": home_possession, "Away possession": away_possession,
                    "Home lineups": home_lineup, "Away lineups": away_lineup,
                   "Home Formation": home_formation, "Away formation": away_formation}

            matches_data.loc[len(matches_data)] = row

            print(f" Home team: {home_team}, {home_odds}, {home_score}, {home_SOT}, {home_xG}, {home_possession}, {home_rating}, {home_formation}, {len(home_lineup)}")
            print(f" away team: {away_team}, {away_odds}, {away_score}, {away_SOT}, {away_xG}, {away_possession}, {away_rating}, {away_formation}, {len(away_lineup)}")

            # When we get done with a match, we'll close its window and move back to the main window

            driver.close()
            driver.switch_to.window(windows[0])

        except Exception as e:

            print(e)
            windows = driver.window_handles

            # We have to get back, in case of error, to our original window

            driver.close()
            driver.switch_to.window(windows[0])


    driver.quit()

    # We'll save our dataframe in a csv file
    # we'll call it for the seasons that we want and store them in the data directory

    matches_data.to_csv(f'data/{season_name}.csv')

    return

# And finally, we'll test our function with the La Liga and Ligue 1 seasons:

liga_seasons = {
    'Liga23-24': "https://www.flashscore.com/football/spain/laliga-2023-2024/#/SbZJTabs",
    'Liga22-23': "https://www.flashscore.com/football/spain/laliga-2022-2023/#/COQ6iu30",
    'Liga21-22': "https://www.flashscore.com/football/spain/laliga-2021-2022/#/MPV5cuep",
    'Liga20-21': "https://www.flashscore.com/football/spain/laliga-2020-2021/#/I58n6IRP",
}

ligue1_seasons = {
    'Ligue1_22-23': "https://www.flashscore.com/football/france/ligue-1-2022-2023/#/zmkW5aIi",
    'Ligue1_23-24': "https://www.flashscore.com/football/france/ligue-1-2023-2024/#/Q1sSPOn5",
    'Ligue1_21-22': "https://www.flashscore.com/football/france/ligue-1-2021-2022/#/0W4LIGb1",
    'Ligue1_20-21': "https://www.flashscore.com/football/france/ligue-1-2020-2021/#/6upiPpqU"
}

scrape_season("Ligue1_21-22", ligue1_seasons["Ligue1_21-22"])