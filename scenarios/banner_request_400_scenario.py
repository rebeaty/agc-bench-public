"""HELM Scenario: BannerRequest400 foreground blueprint for rendered banners."""

import json
import os
import urllib.request
import zipfile
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class BannerRequest400Scenario(Scenario):
    name = "banner_request_400"
    description = "sony/BannerAgency"
    tags = ["creativity", "design", "layout", "multimodal"]

    REPO_URL = "https://github.com/sony/BannerAgency/archive/refs/heads/main.zip"

    TASK_PROMPT = (
        "You are BannerAgency's foreground designer for a 300x250 banner. "
        "Given the advertiser logo image and banner request, create a precise "
        "JSON layout blueprint that can be deterministically rendered into the "
        "final banner image.\n\n"
        "Use one of the official BannerAgency layout styles such as left-content, "
        "right-content, z-pattern, f-pattern, centered, rule-of-thirds, "
        "diagonal, asymmetrical, top-down, pyramid, or grid.\n\n"
        "Implementation guidance from the released prompt:\n"
        "- Canvas: 300x250 pixels\n"
        "- Headline: 24-32px\n"
        "- Subheadline: 18-24px\n"
        "- Body text: 14-16px\n"
        "- CTA: 16-18px\n"
        "- Aim for 70-90% spatial utilization\n"
        "- Reserve at least 10px of clear space from all edges\n"
        "- Keep strong contrast and readable hierarchy\n"
        "- Logo width: about 15-20% of banner width while preserving aspect ratio\n"
        "- Include a clear CTA button when appropriate\n"
        "- Specify all x/y positions and widths/heights in absolute pixels\n\n"
        "Return a JSON object with this schema exactly:\n"
        "{{\n"
        '  "background_color": "#RRGGBB",\n'
        '  "elements": [\n'
        "    {{\n"
        '      "type": "headline|subheadline|body|cta_button|logo|shape",\n'
        '      "text": "text content or empty string",\n'
        '      "position": {{"x": 0, "y": 0}},\n'
        '      "size": {{"width": 0, "height": 0}},\n'
        '      "font_size": 0,\n'
        '      "font_weight": "regular|bold",\n'
        '      "text_color": "#RRGGBB",\n'
        '      "fill_color": "#RRGGBB",\n'
        '      "corner_radius": 0,\n'
        '      "align": "left|center|right"\n'
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        "Banner request: {banner_request}\n"
        "Target audience: {target_audience}\n"
        "Primary purpose: {primary_purpose}\n\n"
        "Use the uploaded logo image for brand identity cues. Output valid JSON only."
    )

    def _download_data(self, output_path: str) -> str:
        """Download and extract the BannerAgency repository."""
        os.makedirs(output_path, exist_ok=True)
        data_dir = os.path.join(output_path, "BannerAgency-main", "BannerRequest400")
        if os.path.exists(data_dir):
            return data_dir

        # Also check a shared cache so multiple runs don't re-download the 16MB zip.
        shared_cache = os.path.join("benchmark_output", "scenarios", "banner_request_400")
        shared_data = os.path.join(shared_cache, "BannerAgency-main", "BannerRequest400")
        if os.path.exists(shared_data):
            return shared_data

        zip_path = os.path.join(output_path, "BannerAgency.zip")
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(self.REPO_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)

        return data_dir

    def get_instances(self, output_path: str):
        data_dir = self._download_data(output_path)

        # Load concrete requests
        with open(os.path.join(data_dir, "concrete_5k.json")) as f:
            advertisers = json.load(f)

        logo_dir = os.path.join(data_dir, "logos_png")

        instances = []
        for adv in advertisers:
            logo_path = os.path.join(logo_dir, adv["logo_name"])

            for pair_key in ["pair_1", "pair_2", "pair_3", "pair_4"]:
                pair = adv["advertising_variations"][pair_key]

                prompt_text = self.TASK_PROMPT.format(
                    banner_request=pair["concrete_request_300x250"],
                    target_audience=pair["target_audience"],
                    primary_purpose=pair["primary_purpose"],
                )

                # Multimodal: logo image + text prompt
                multimedia_content = MultimediaObject([
                    MediaObject(
                        content_type="text/plain",
                        text=prompt_text,
                    ),
                    MediaObject(
                        content_type="image/png",
                        location=logo_path,
                    ),
                ])

                instance_id = f"{adv['id']:03d}_{pair_key}"

                # Open-ended generation: no gold reference
                instances.append(Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=[],
                    split=TEST_SPLIT,
                    id=instance_id,
                    extra_data={
                        "advertiser": adv["advertiser"],
                        "logo_path": logo_path,
                        "logo_description": adv["logo_description"],
                        "banner_request": pair["concrete_request_300x250"],
                        "target_audience": pair["target_audience"],
                        "primary_purpose": pair["primary_purpose"],
                    },
                ))

        return instances
