import os
import json
import time
import sys

if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

EMAIL = "tatanrata2@gmail.com"
PASSWORD = "doctordoom00"


def scrapping(influencer: str):
    os.makedirs("data", exist_ok=True)  

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 720}
            )
        page = context.new_page()
        page.set_extra_http_headers({
    "accept-language": "en-US,en;q=0.9"
})


        try:
            page.goto("https://www.linkedin.com/login")
            page.fill("input#username", EMAIL)
            page.fill("input#password", PASSWORD)
            page.click("button[type='submit']")
            page.wait_for_load_state("domcontentloaded")
            time.sleep(5)

            activity_url = f"https://www.linkedin.com/in/{influencer}/recent-activity/all/"
            page.goto(activity_url)
            page.wait_for_load_state("domcontentloaded")
            time.sleep(5)

            for _ in range(3):
                page.mouse.wheel(0, 2000)
                time.sleep(2)

            posts = page.locator("div.feed-shared-update-v2")
            count = posts.count()

            post_content = []

            for i in range(count):
                post = posts.nth(i)
                try:
                    text = post.locator("span[dir='ltr']").all_inner_texts()
                    content = " ".join(text).strip() or "[No text found]"
                except PlaywrightTimeout:
                    content = "[No text found]"

                try:
                    likes = post.locator("span.social-details-social-counts__reactions-count").first.inner_text().strip()
                except:
                    likes = "[likes not found]"

                try:
                    span_texts = post.locator("span").all_inner_texts()
                    comment_spans = [t for t in span_texts if "comment" in t.lower()]
                    comments = comment_spans[0] if comment_spans else "[comments not found]"
                except:
                    comments = "[comments not found]"

                post_content.append({
                    "post_number": i + 1,
                    "content": content,
                    "likes": likes,
                    "comments": comments
                })

            file_path = os.path.join("data", f"{influencer}_posts.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({"posts": post_content}, f, indent=4, ensure_ascii=False)


        except Exception as e:
            print(f"❌ Error occurred: {e}")

        finally:
            browser.close()
