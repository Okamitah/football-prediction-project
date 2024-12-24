# This function will take more variables than the last one, with most of the code staying the same
# And we'll apply it to multiple more seasons from flashscore

import time

import pandas as pd

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# I opted to make a function that scrapes a whole season, with a given link, to organise our data how it suits us
# Whether it is to work on multiple leagues in a single season, or a single league during multiple seasons

def scrape_season2(season_name, season_link):

    # And initialise the Selenium Webdriver

    service = Service()
    options = webdriver.ChromeOptions()
    options.binary_location = "/snap/bin/chromium"
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=9222")
    driver = webdriver.Chrome(service=service, options=options)
    driver.maximize_window()

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
            time.sleep(2)

            # Now we'll pick our features:

            home_team = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//a[@class='participant__participantName participant__overflow '])[1]"))).text

            away_team = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//a[@class='participant__participantName participant__overflow '])[2]"))).text

            home_score = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='detailScore__wrapper']/span)[1]"))).text

            away_score = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='detailScore__wrapper']/span)[3]"))).text

            home_odds = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='oddsValueInner'])[1]"))).text

            draw_odds = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='oddsValueInner'])[2]"))).text

            away_odds = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='oddsValueInner'])[3]"))).text

            # xG is the expected goals

            home_xG = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-homeValue_-iJBW'])[1]"))).text

            home_possession = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-homeValue_-iJBW'])[2]"))).text

            # SOT is the shots on target

            home_SOT = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-homeValue_-iJBW'])[3]"))).text

            away_xG = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-awayValue_rQvxs'])[1]"))).text

            away_possession = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-awayValue_rQvxs'])[2]"))).text

            away_SOT = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "(//div[@class='wcl-value_IuyQw wcl-awayValue_rQvxs'])[3]"))).text

            # Now we'll move to the lineups section of the match, so we get the ratings, formations and lineups:

            lineups_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Lineups')]")
            driver.maximize_window()

            #Sometimes these buttons are stubborn, so we have to call in the big guns: ActionChains

            actions.move_to_element(lineups_button).perform()
            lineups_button.click()

            # We'll let our driver chill for a bit

            driver.execute_script("window.scrollBy(0, window.innerHeight / 2);")
            time.sleep(2)

            home_rating = WebDriverWait(driver, 30).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='wcl-caption_xZPDJ wcl-scores-caption-03_LG4YJ wcl-bold_slHaC'])[1]"))).text

            away_rating = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='wcl-caption_xZPDJ wcl-scores-caption-03_LG4YJ wcl-bold_slHaC'])[2]"))).text

            home_formation = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.XPATH, "(//span[@class='lf__headerPart'])[1]"))).text

            away_formation = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
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

    matches_data.to_csv(f'data2/{season_name}.csv')

    return

# And finally, we'll test our function with the La Liga and Ligue 1 seasons:

liga_seasons = {
    'Liga09-10': "https://www.flashscore.com/football/spain/laliga-2009-2010/#/U5sTpd2G",
    'Liga10-11': "https://www.flashscore.com/football/spain/laliga-2010-2011/#/E5ntFgfd",
    'Liga11-12': "https://www.flashscore.com/football/spain/laliga-2011-2012/#/2ar1uWVA",
    'Liga12-13': "https://www.flashscore.com/football/spain/laliga-2012-2013/#/lCDqolol/table/overall",
    'Liga13-14': "https://www.flashscore.com/football/spain/laliga-2013-2014/#/0M9LMq9C/table/overall",
    'Liga14-15': "https://www.flashscore.com/football/spain/laliga-2014-2015/#/EuKuM5MI/table/overall",
    'Liga15-16': "https://www.flashscore.com/football/spain/laliga-2015-2016/#/IclHToOB/table/overall",
    'Liga16-17': "https://www.flashscore.com/football/spain/laliga-2016-2017/#/rH7SFawI/table/overall",
    'Liga17-18': "https://www.flashscore.com/football/spain/laliga-2017-2018/#/8COD1Gpp",
    'Liga18-19': "https://www.flashscore.com/football/spain/laliga-2018-2019/#/IVm2O3QA/table/overall",
    'Liga23-24': "https://www.flashscore.com/football/spain/laliga-2023-2024/#/SbZJTabs",
    'Liga22-23': "https://www.flashscore.com/football/spain/laliga-2022-2023/#/COQ6iu30",
    'Liga21-22': "https://www.flashscore.com/football/spain/laliga-2021-2022/#/MPV5cuep",
    'Liga20-21': "https://www.flashscore.com/football/spain/laliga-2020-2021/#/I58n6IRP",
    'Liga19-20': "https://www.flashscore.com/football/spain/laliga-2019-2020/#/MNGIgau5"
}

ligue1_seasons = {
    'Ligue1_09-10': "",
    'Ligue1_10-11': "",
    'Ligue1_11-12': "",
    'Ligue1_12-13': "",
    'Ligue1_13-14': "",
    'Ligue1_14-15': "",
    'Ligue1_15-16': "",
    'Ligue1_16-17': "",
    'Ligue1_17-18': "",
    'Ligue1_18-19': "",
    'Ligue1_19-20': "",
    'Ligue1_22-23': "https://www.flashscore.com/football/france/ligue-1-2022-2023/#/zmkW5aIi",
    'Ligue1_23-24': "https://www.flashscore.com/football/france/ligue-1-2023-2024/#/Q1sSPOn5",
    'Ligue1_21-22': "https://www.flashscore.com/football/france/ligue-1-2021-2022/#/0W4LIGb1",
    'Ligue1_20-21': "https://www.flashscore.com/football/france/ligue-1-2020-2021/#/6upiPpqU"
}

serieA_seasons = {
    'SerieA20-21': "https://www.flashscore.com/football/italy/serie-a-2020-2021/#/hKAgCv61",
    'SerieA21-22': "https://www.flashscore.com/football/italy/serie-a-2021-2022/#/YHxmuFsJ",
    'SerieA22-23': "https://www.flashscore.com/football/italy/serie-a-2022-2023/#/UcnjEEGS",
    'SerieA23-24': "https://www.flashscore.com/football/italy/serie-a-2023-2024/#/GK3TOCxh",
    'SerieA19-20': "https://www.flashscore.com/football/italy/serie-a-2019-2020/#/pImv7QRb"
}

bundelsiga_seasons = {
    'Bundes19-20': "https://www.flashscore.com/football/germany/bundesliga-2019-2020/#/dAfCUJq0",
    'Bundes20-21': "https://www.flashscore.com/football/germany/bundesliga-2020-2021/#/bk1Zgnfk",
    'Bundes21-22': "https://www.flashscore.com/football/germany/bundesliga-2020-2021/#/bk1Zgnfk",
    'Bundes22-23': "https://www.flashscore.com/football/germany/bundesliga-2022-2023/#/OIbxfZZI",
    'Bundes23-24': "https://www.flashscore.com/football/germany/bundesliga-2023-2024/#/OWq2ju22"
}

pl_seasons = {
    'PL09-10': "https://www.flashscore.com/football/england/premier-league-2009-2010/#/COuk57Ci",
    'PL10-11': "https://www.flashscore.com/football/england/premier-league-2010-2011/#/S0jBn4Ad",
    'PL11-12': "https://www.flashscore.com/football/england/premier-league-2011-2012/#/xOR3gQta",
    'PL12-13': "https://www.flashscore.com/football/england/premier-league-2012-2013/#/2wDJRkib",
    'PL13-14': "https://www.flashscore.com/football/england/premier-league-2013-2014/#/Uc3Ascea",
    'PL14-15': "https://www.flashscore.com/football/england/premier-league-2014-2015/#/U9wG88N6",
    'PL15-16': "https://www.flashscore.com/football/england/premier-league-2015-2016/#/faBBhyuM",
    'PL16-17': "https://www.flashscore.com/football/england/premier-league-2016-2017/#/fZHsKRg9",
    'PL17-18': "https://www.flashscore.com/football/england/premier-league-2017-2018/#/WOO1nDO2",
    'PL18-19': "https://www.flashscore.com/football/england/premier-league-2018-2019/#/v1t6uXL7",
    'PL19-20': "https://www.flashscore.com/football/england/premier-league-2019-2020/#/CxZEqxa7",
    'PL20-21': "https://www.flashscore.com/football/england/premier-league-2020-2021/#/zTRyeuJg",
    'PL21-22': "https://www.flashscore.com/football/england/premier-league-2021-2022/#/6kJqdMr2",
    'PL22-23': "https://www.flashscore.com/football/england/premier-league-2022-2023/#/nunhS7Vn",
    'PL23-24': "https://www.flashscore.com/football/england/premier-league-2023-2024/#/I3O5jpB2"
}

primeira_seasons = {
    'PR19-20': "https://www.flashscore.com/football/portugal/liga-portugal-2019-2020/#/r91KaWFr",
    'PR20-21': "https://www.flashscore.com/football/portugal/liga-portugal-2020-2021/#/6mtPHHfM",
    'PR21-22': "https://www.flashscore.com/football/portugal/liga-portugal-2021-2022/#/SYQdiOke",
    'PR22-23': "https://www.flashscore.com/football/portugal/liga-portugal-2022-2023/#/0npsaei3",
    'PR23-24': "https://www.flashscore.com/football/portugal/liga-portugal-2023-2024/#/xQ60fbmB",
}

jupiler_seasons = {
    'JL19-20': "https://www.flashscore.com/football/belgium/jupiler-pro-league-2019-2020/#/QwUfxnS7",
    'JL20-21': "https://www.flashscore.com/football/belgium/jupiler-pro-league-2020-2021/#/lxRJcNej",
    'JL21-22': "https://www.flashscore.com/football/belgium/jupiler-pro-league-2021-2022/#/88GNJ4I9",
    'JL22-23': "https://www.flashscore.com/football/belgium/jupiler-pro-league-2022-2023/#/v3vlzrwf",
    'JL23-24': "https://www.flashscore.com/football/belgium/jupiler-pro-league-2023-2024/#/G6IvtOdO",
}

eredivisie_seasons = {
    'ED19-20': "https://www.flashscore.com/football/netherlands/eredivisie-2019-2020/#/Ym9YtRCF",
    'ED20-21': "https://www.flashscore.com/football/netherlands/eredivisie-2020-2021/#/2D6NTKwA",
    'ED21-22': "https://www.flashscore.com/football/netherlands/eredivisie-2021-2022/#/SfQjVhXC",
    'ED22-23': "https://www.flashscore.com/football/netherlands/eredivisie-2022-2023/#/CfNLdj8j",
    'ED23-24': "https://www.flashscore.com/football/netherlands/eredivisie-2023-2024/#/zeqqyRgJ",
}

turkiyesl_seasons = {
    'TSL19-20': "https://www.flashscore.com/football/turkey/super-lig-2019-2020/#/A146aXLs",
    'TSL20-21': "https://www.flashscore.com/football/turkey/super-lig-2020-2021/#/4Q6NO4a2",
    'TSL21-22': "https://www.flashscore.com/football/turkey/super-lig-2021-2022/#/zLsIuMTj",
    'TSL22-23': "https://www.flashscore.com/football/turkey/super-lig-2022-2023/#/lh8AshXk",
    'TSL23-24': "https://www.flashscore.com/football/turkey/super-lig-2023-2024/#/KzRFDJ4U",
}

botola_seasons = {
    'Botola19-20': "https://www.flashscore.com/football/morocco/botola-pro-2019-2020/#/UoFOMiam",
    'Botola20-21': "https://www.flashscore.com/football/morocco/botola-pro-2020-2021/#/O46wsvs5",
    'Botola21-22': "https://www.flashscore.com/football/morocco/botola-pro-2021-2022/#/S0NSiyWK",
    'Botola22-23': "https://www.flashscore.com/football/morocco/botola-pro-2022-2023/#/GKo7VnUn",
    'Botola23-24': "https://www.flashscore.com/football/morocco/botola-pro-2023-2024/#/xEMmpc7l",
}

for season in turkiyesl_seasons:
    scrape_season2(season, turkiyesl_seasons[season])