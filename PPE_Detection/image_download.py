# pip install simple_image_download==0.4
from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

keywords = ["building workers with mask"]

for kw in keywords:
    response().download(kw, 200)