# Building a basic web scraper from known URL, using this tutorial:
# https://web.stanford.edu/~zlotnick/TextAsData/Web_Scraping_with_Beautiful_Soup.html

# You have to know explicitly what you are looking for. We know we want the list of athlete names,
# their graduation dates, and the universities they will be going to. But where on the page is that contained?
# In this case, the male and female athletes are all contained in two divs with the name "content-wrap"

from bs4 import BeautifulSoup
import urllib2
import pandas as pd
import re
from fuzzywuzzy import fuzz

# Load in conents of page as urllib2 object of some sort
page = urllib2.urlopen('http://www.newhampton.org/Page/Athletics/Alumni-in-Athletics')

# Convert page to soup knowing that the page is in html
soup = BeautifulSoup(page, 'html.parser')

content = soup.find_all('div', class_='content-wrap')
# This website uses generic naming for each division, which means it's impossible to directly identify where the
# athlete information is from the div name alone. There are 13 'content-wrap' divs, and 5 is for males, 6 for females
male_athletes = content[5]
female_athletes = content[6]

# Now we need to parse out the useful information. It is consistently structured (I hope) with a <b> .. </b> around
# each entry, so that should be enough to get us a list of names.
# NOTE: BeautifulSoup can parse out HTML pieces (e.g. <b>, </b>) easily, so let it do that first
# The .get_text(',') returns a single string with comma-separated entries buried in it, and the .split(',') will
# split those out into a list

list_males = male_athletes.get_text('\t').split('\t')
list_females = female_athletes.get_text('\t').split('\t')

# Known pieces that are returned from the beautifulsoup processing that are not helpful
known_unwanted = ['\n', ' ']

l_m = [x for x in list_males if x not in known_unwanted]
l_f = [x for x in list_females if x not in known_unwanted]

# Build dataframe from the elements in the list
athletes = pd.DataFrame(columns=['sport', 'name', 'year', 'college', 'gender'])
curr_sport = None
for i, entry in enumerate(l_m + l_f):
    entry = re.sub('\xa0|\xc2', ' ', entry)  # remove any non-ascii whitespace
    if i < len(l_m):
        gender = 'M'
    else:
        gender = 'F'
    # try:
    #     print 'processing entry ({i}) {e}'.format(i=i, e=entry)
    # except UnicodeEncodeError:
    #     print 'processing entry ({i})'.format(i=i)

    if len(entry.split(' ')) <= 2:
        # Because some colleges are incorrectly formatted, the beautifulsoup returns split entries between
        # athlete and college. This causes the college to be recognized as the sport, thereby influencing all
        # downstream athletes. Trying to avoid this.
        # if ['University', 'College'] in curr_sport:
        if any([x in entry for x in ['University', 'College']]):
            print '({i}) curr_sport listing: {s}. Skipping'.format(i=i, s=entry)
            continue
        else:
            curr_sport = entry
    else:
        try:
            # We know there will be a year (2013) after the name, so leverage that
            name = entry.split('(')[0].rstrip().lstrip()
            # Find digits between ( ), strip off () and convert to int
            year = int(re.findall('\(\d+\)', entry)[0].lstrip('(').rstrip(')'))
            # split only where there are digits surrounded by parentheses, then parse out the college,
            # use regex to replace a dash OR any special characters created by the conversion to UTF8 in
            # case the coding was different across the entries (which it is). Then strip off all the extra
            # whitespace
            college = re.sub('\xa0|\xc2|-|\xe2|\x80|\x93|\x94', '', re.split('\(\d+\)', entry)[-1].encode('utf-8')).lstrip().rstrip()
            # There are some situations where there is a line break between school year and college because
            # the college is surrounded by <span> (WHY?@!?!). How to handle this??
            if len(college) == 0:
                print '({i}) Couldnt parse college. Skipping'.format(i=i)
                continue

            athletes = athletes.append({'sport': curr_sport,
                             'name': name,
                             'year': year,
                             'college': college,
                             'gender': gender},
                            ignore_index=True)
        except IndexError:
            print '({i}) {e} \nEntry not formatted properly. Skipping...'.format(e=entry, i=i)
            continue

athletes = athletes.reset_index(drop=True)
athletes.year = athletes.year.astype(int)  # for some reason inserting via {} forces conversion to float??
# print athletes

# athletes.to_csv('test_parse_athletes.csv', index=False)

# Now read in NCAA info from http://web1.ncaa.org/onlineDir/exec2/divisionListing?sortOrder=0&division=1

# page = urllib2.urlopen('http://web1.ncaa.org/onlineDir/exec2/divisionListing?sortOrder=0&division=1')
# NCAA page required additional header info in the request

divisions = [1, 2, 3]
for division in divisions:
    print 'processing NCAA division {}'.format(division)
    site = 'http://web1.ncaa.org/onlineDir/exec2/divisionListing?sortOrder=0&division={}'.format(division)
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8',
           'Connection': 'keep-alive'}

    req = urllib2.Request(site, headers=hdr)

    try:
        page = urllib2.urlopen(req)
    except urllib2.HTTPError, e:
        print e.fp.read()

    soup = BeautifulSoup(page, 'html.parser')

    # content = soup.find_all('tr')[2:]
    # First couple of 'tr' entries are metadata - not useful

    rows = list()
    for tr in soup.find_all('tr')[1:]:
        tds = tr.find_all('td')
        rows.append(tds)

    # Know that row[0] = headers, row[1] = blank line, row[2:] is the list of schools
    for i, row in enumerate(rows):
        # Each row contains a list of entries, one per <td> </td> pair. Use this format
        if (i == 0) & (division == 1):
            # Establish the dataframe
            headers = [str(x.find_all('b')[0]).lstrip('<b>').rstrip('</b>') for x in row]
            ncaa = pd.DataFrame(columns=headers)
        elif (i==0) & (division > 1):
            continue
        elif i == 1:
            continue
        else:
            row_content = list()
            # Push data that is not empty into the dataframe
            for i, entry in enumerate(row):
                if i < 2:
                    # University or Conference entries, which are links
                    try:
                        row_content.append(str(entry.find_all('a')[0]).split('</')[0].split('>')[-1].replace('&amp;', '&'))
                    except IndexError:
                        print 'Couldnt process Univ or Conference. Adding blank.'
                        row_content.append('')
                else:
                    # Div, Re-class, and State listing
                    try:
                        row_content.append(re.findall('\s\w+-\w+\s|\s\w+\s', str(row[i]))[0].lstrip().rstrip())
                        # row_content.append(re.findall('\s^[\w+]+(-\w+)\s', str(row[2])) ) # conference only
                        # row_content.append(re.findall('\s\w+\s', str(row[2]))[0].lstrip().rstrip()) # state only
                    except IndexError:
                        # When there is no reclass entry, will throw this index error
                        row_content.append('')
            ncaa = ncaa.append(dict(zip(headers, row_content)), ignore_index=True)

print ncaa
ncaa.to_csv('CSV/ncaa.csv', index=False)

# ## Merge the college info and the NCAA info in one final dataframe
athlete_conf = pd.DataFrame(columns=[athletes.columns.union(ncaa.columns.union(['full_prob', 'part_prob']))])
for idx in athletes.index:
    athlete = athletes.loc[idx]
    college = athlete.college
    # a little pre-processing to make sure generic words like 'college' or 'university' are not matched on
    x = re.sub('College|University|college|university|univ', '', college).lstrip().rstrip()
    full_probs = list()
    part_probs = list()
    for inst in ncaa.Institution:
        y = re.sub('College|University|college|university|univ', '', inst).lstrip().rstrip()
        full_probs.append(fuzz.ratio(x, y))
        part_probs.append(fuzz.partial_ratio(x, y))

    max_prob = max(full_probs)
    ncaa_entry = ncaa.ix[full_probs.index(max_prob)]
    athlete['full_prob'] = max_prob
    athlete['part_prob'] = part_probs[full_probs.index(max_prob)]

    athlete_conf = athlete_conf.append(athlete.append(ncaa_entry), ignore_index=True)

print athlete_conf.head()

athlete_conf = athlete_conf.rename(columns={'Institution': 'ncaa_college', 'Division': 'ncaa_div', 'Conference': 'ncaa_conf',
                             'Reclass Division': 'ncaa_reclass', 'State': 'ncaa_state'})
col_order = ['sport', 'name', 'year', 'gender', 'college', 'ncaa_college', 'full_prob', 'part_prob', 'ncaa_conf',
             'ncaa_div', 'ncaa_reclass', 'ncaa_state']
athlete_conf.to_csv('CSV/athlete_ncaa_prob_matches.csv', columns=col_order)
