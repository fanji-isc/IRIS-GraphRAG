from diskcache import Cache

# Path should match the one used in your app
cache = Cache('./llm_cache')

cache.clear()