import requests
import json
import numpy as np
import uuid
from tqdm import tqdm
from const import *

class img():
    def __init__(self, data):
        self.id = str(uuid.uuid4())
        self.name = f"{self.id}.png"
        f = open(f"{PIC_FOLDER}/{self.name}", "wb")
        f.write(data)
        f.close()

class GM_API():
    
    def __init__(self):
        self.endpoint = "https://maps.googleapis.com/maps/api/streetview"
        self.params = {}
        return

    def fetch(self):
        res = requests.get(self.endpoint, params=self.params)

        if res.status_code != 200:
            return False
        else:
            return True

class streetview(GM_API):

    def __init__(self, lat, lng, sizeX=PIC_HEIGHT, sizeY=PIC_WIDTH, fov=120, heading=0, radius=400):
        self.endpoint = "https://maps.googleapis.com/maps/api/streetview"
        self.params = {
            "location": f"{lat},{lng}",
            "size": f"{sizeX}x{sizeY}",
            "fov": fov,
            "heading": heading,
            "radius": radius,
            "return_error_code": "true",
            "key": MAP_API
        }
        return

    def fetch(self):
        res = requests.get(self.endpoint, params=self.params)

        if res.status_code != 200:
            return None
        else:
            return img(res.content)

class streetview_metadata(streetview):
    def __init__(self, lat, lng, sizeX=PIC_HEIGHT, sizeY=PIC_WIDTH, fov=120, heading=0, radius=200):
        super().__init__(lat, lng, sizeX, sizeY, fov, heading, radius)
        self.endpoint = "https://maps.googleapis.com/maps/api/streetview/metadata"
        self.result = None

    def fetch(self):
        res = requests.get(self.endpoint, params=self.params)

        if res.status_code != 200:
            return False
        else:
            self.result = res.text
            return True

class geocode(GM_API):
    def __init__(self, lat, lng, lang="zh-TW"):
        self.endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
        self.params = {
            "latlng": f"{lat},{lng}",
            "result_type": "administrative_area_level_1",
            "language": lang,
            "key": MAP_API
        }
        return

    def fetch(self):
        res = requests.get(self.endpoint, params=self.params)

        if res.status_code != 200:
            return False
        else:
            self.result = res.text
            return True

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            data = f.read()
            data_dict = json.loads(data)
            return data_dict
    else:
        return {}

def write_metadata(data_dict):
    with open(METADATA_FILE, "w") as f:
        f.write(json.dumps(data_dict, indent=4, ensure_ascii=False))
    return



def main():

    mds = load_metadata()
    
    success_cnt = 0

    t = tqdm(range(N))

    for i in t:
        latitude = np.around(np.random.uniform(MIN_LATITUDE, MAX_LATITUDE), decimals=10)
        longitude = np.around(np.random.uniform(MIN_LONGITUDE, MAX_LONGITUDE), decimals=10)
        heading = int(np.random.uniform(0, 360))
    
        sv = streetview(latitude, longitude, heading=heading)
        res_img = sv.fetch()

        if res_img is None:
            continue
        
        svmd = streetview_metadata(latitude, longitude)

        if not svmd.fetch():
            print("metadata not found.")
            continue

        mds[res_img.id] = json.loads(svmd.result)
        mds[res_img.id]["id"] = res_img.id
        
        location = mds[res_img.id]["location"]
        gc = geocode(location["lat"], location["lng"])

        if not gc.fetch():
            print("geocode error")
            continue

        gc_dict = json.loads(gc.result)
        try:
            mds[res_img.id]["city"] = gc_dict["results"][0]["address_components"][0]["long_name"]
        except:
            os.remove(f"{PIC_FOLDER}/{res_img.name}")
            del mds[res_img.id]
            print(f"{PIC_FOLDER}/{res_img.name} removed")
            continue

        success_cnt += 1
        t.set_description(f"SuccessRate: {success_cnt}/{N}")
        t.refresh()
    print(f"success rate: {success_cnt}/{N}")
    write_metadata(mds)

if __name__ == '__main__':
    main()
