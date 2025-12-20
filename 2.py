# our "notebook " for storing the mapping of long url to short url
map = {}

# counter = 1, for indicating each url
id = 1

# convert long url to short url
def save_url(long_url):
    global id
    
    #write it in our notebook
    short_id = str(id)
    map[short_id] = long_url 

    #update id
    id += 1

    #Return the short url
    return f"mywebsite/{short_id}"


# get shor url
def get_url(short_id):
    if short_id in map:
        return map[short_id]
    else:
        return "Not found!"
    
    pass


#--------------------TEST---------------------
print("=== Testing our URL shortener === \n")


# Save some urls
url1 = save_url("https://www.google.com")
print(f"saved google as {url1}")

url2 = save_url("https://www.facebook.com")
print(f"saved facebook as {url2}")

url3 = save_url("https://www.amazon.com")
print(f"saved amazon as {url3}")

# Get long urls
print(f"mywebsite/1  -> {get_url('1')}")
print(f"mywebsite/2 -> {get_url('2')}")
print(f"mywebsite/3 -> {get_url('3')}")
















