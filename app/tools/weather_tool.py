import aiohttp


async def get_weather(location: str) -> dict:
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    geocode_params = {
        "name": location,
        "count": 1,
        "language": "zh",
        "format": "json",
    }
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    # ★ 一个 session 复用两次请求
    async with aiohttp.ClientSession() as session:
        # ── 1. 地理编码：城市名 → 经纬度 ──
        async with session.get(geocode_url, params=geocode_params) as resp:
            geo_data = await resp.json()

        results = geo_data.get("results", [])
        if not results:
            raise ValueError(f"未找到地点: {location}")

        first = results[0]
        latitude = first["latitude"]
        longitude = first["longitude"]
        resolved_name = first["name"]
        country = first.get("country", "")
        admin1 = first.get("admin1", "")

        # ── 2. 查询当前天气 ──
        forecast_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }
        async with session.get(forecast_url, params=forecast_params) as resp:
            forecast_data = await resp.json()

        current = forecast_data.get("current")
        units = forecast_data.get("current_units")

        return {
            "locationQueried": location,
            "resolvedLocation": {
                "name": resolved_name,
                "admin1": admin1,
                "country": country,
                "latitude": latitude,
                "longitude": longitude,
            },
            "current": current,
            "units": units,
            "source": "open-meteo.com",
        }