#!/usr/bin/env python3
"""Accuracy question set (eval_v1) for comparing decision models.

The speed benchmark (bench2.py) uses 10 tickets whose gold labels are ambiguous
("complaint" vs "urgent issue"), so it cannot tell models apart. This set has 330 items whose
answer is clear to a person, in seven tasks:

  task     items  image  pool  what is asked
  intent    180   no      60   which request a support ticket makes: 120 plain (2 per request type),
                               30 in Japanese, 30 that mention other topics or say what they are not asking
  shape      30   yes     48   which coloured shape is drawn ("a red circle")
  count      20   yes     30   how many black dots there are
  word       30   yes     60   which word is written on a sign
  error      20   yes     30   which problem an error-dialog screenshot shows (the text alone does not say)
  scene      30   yes     48   which of four objects is in a given corner
  receipt    20   yes     60+  which amount on a receipt is the total (cash paid is larger)

Each item becomes a single `choice` question with K options: the gold label plus K-1 distractors,
chosen with a fixed seed so everyone gets the same questions. Confusable labels come first
(item["near"]: requests of the same kind, the other objects in the picture, the other amounts on the
receipt), then random labels from the task's pool. K can be at most 30.

Build the files (images are drawn once and stored, so fonts do not matter later):
  python bench/evalset.py dataset/eval_v1
Use from code:
  from evalset import load_items, make_question
"""
import json, pathlib, random, sys

INTENTS = {
    "reset a forgotten password": [
        "I forgot my password and the reset link never arrives. How do I get back in?",
        "Can't remember my password. Please help me set a new one.",
    ],
    "change the account email address": [
        "I'm leaving my company, so I need my login email changed to my personal gmail.",
        "How do I switch the email address on my account to a new one?",
    ],
    "turn on two-factor authentication": [
        "I'd like to add a second login step with an authenticator app. Where is that setting?",
        "Our security team requires 2FA for every account. How do I enable it?",
    ],
    "unlock an account locked after failed logins": [
        "I typed the wrong password a few times and now it says my account is locked for 24 hours. Can you unlock it?",
        "Account locked out after too many attempts, I need access today.",
    ],
    "close and delete the account": [
        "I don't use your service anymore. Please close my account permanently.",
        "How can I delete my account completely? I won't be coming back.",
    ],
    "report a login by someone else": [
        "I got an alert that someone signed in to my account from Brazil. That wasn't me!",
        "There are sessions in my account history from devices I have never owned.",
    ],
    "refund a duplicate charge": [
        "You billed my card twice for the same March order. Please return one of the payments.",
        "I see two identical charges of $49 on the same day. I only bought once.",
    ],
    "update the saved payment card": [
        "My credit card expired, I need to put the new card on file before the next renewal.",
        "How do I replace the card you charge every month with a different one?",
    ],
    "send a copy of an invoice": [
        "Our accountant needs the invoice for the August payment. Can you send me a copy?",
        "Where can I download a PDF of last month's invoice?",
    ],
    "change the billing address": [
        "We moved offices. Please update the address that appears on our invoices.",
        "The billing address on my account is old, how do I edit it?",
    ],
    "explain the price of each plan": [
        "What's the difference in cost between the Basic, Pro and Business plans?",
        "How much does each of your plans cost per month?",
    ],
    "upgrade to a higher plan": [
        "We've outgrown the starter tier. I want to move up to Pro today.",
        "Please switch our team to the Business plan, we need more seats.",
    ],
    "downgrade to a cheaper plan": [
        "We don't use the advanced features. Can we move down to the basic tier to save money?",
        "Please switch me from Premium to the cheaper Standard plan next cycle.",
    ],
    "cancel the subscription": [
        "Please stop my monthly subscription, I don't want to be charged again.",
        "I want to end my membership before it renews on the 1st.",
    ],
    "pause the subscription for a while": [
        "I'm travelling for two months. Can I put my membership on hold and resume later without losing my data?",
        "Is it possible to freeze my plan over the summer and restart in September?",
    ],
    "fix a promo code that is not accepted": [
        "The code SPRING20 from your newsletter says 'invalid' at checkout.",
        "Your discount coupon isn't applying to my cart even though it hasn't expired.",
    ],
    "redeem a gift card": [
        "My aunt gave me one of your gift cards. Where do I enter the number to use it?",
        "How do I add the balance of a gift card to my account?",
    ],
    "check where an order is": [
        "I ordered on Monday. Has it shipped yet? I don't have a tracking number.",
        "Can you tell me the current status of order #55120?",
    ],
    "complain about a late delivery": [
        "Delivery was promised for last Friday and it's now Thursday. Still nothing. Very disappointed.",
        "My parcel is a week past the guaranteed arrival date.",
    ],
    "change the delivery address of an order": [
        "I just placed an order but used my old address. Can you send it to my new apartment instead?",
        "Please redirect order #8812 to my office address before it ships.",
    ],
    "report an item damaged in shipping": [
        "The box arrived crushed and the glass vase inside was broken.",
        "My new monitor came with a cracked screen, the packaging was torn open.",
    ],
    "report receiving the wrong item": [
        "I ordered the blue backpack but you sent me a black suitcase.",
        "The package contained someone else's order, not the headphones I bought.",
    ],
    "return an item for a refund": [
        "The jacket doesn't suit me. How do I send it back and get my money back?",
        "I'd like to return the unopened coffee machine for a full refund.",
    ],
    "exchange an item for another size": [
        "The shoes are too small. Can I swap them for a size 43?",
        "Could I exchange this medium shirt for a large one?",
    ],
    "report an item missing from the package": [
        "My order should have had three books but only two were in the box.",
        "The charger listed on the packing slip wasn't included with the laptop.",
    ],
    "claim warranty for a product that stopped working": [
        "My blender, bought 8 months ago, suddenly won't turn on at all. Is this covered?",
        "The headphones stopped charging after five months of normal use. I'd like a repair or replacement.",
    ],
    "ask if a product is compatible": [
        "Will your USB-C dock work with a 2019 MacBook Air?",
        "Does this replacement filter fit the model X200 air purifier?",
    ],
    "get the user manual": [
        "I lost the instruction booklet for my rice cooker. Is there a PDF somewhere?",
        "Where can I find setup instructions for the model T5 thermostat?",
    ],
    "report the app crashing on start": [
        "Since yesterday's update the Android app closes immediately every time I open it.",
        "The iPhone app shows the logo for a second and then crashes back to the home screen.",
    ],
    "report the website being slow": [
        "Your dashboard takes almost a minute to load each page today. Everything works, just painfully slowly.",
        "Pages on your site are really sluggish this week, searches take ages to return.",
    ],
    "report the website being completely down": [
        "Your whole website returns 'service unavailable', nobody in our office can reach it.",
        "The site won't load at all from any browser or network. Are you having an outage?",
    ],
    "report a broken link": [
        "The 'Pricing' link in your footer leads to a 404 page.",
        "The download button on the help article points to a page that doesn't exist.",
    ],
    "suggest a new feature": [
        "It would be great if the calendar could export to Google Calendar. Please consider adding that.",
        "Could you add a dark mode to the desktop app? Many of us would love it.",
    ],
    "praise the product": [
        "Just wanted to say your app made our team's planning so much easier. Great job!",
        "Love the latest version, it's fast and beautiful. Thanks to everyone who built it.",
    ],
    "complain about a support agent": [
        "The person I chatted with yesterday was rude and closed the chat without answering me.",
        "Your phone agent hung up on me twice and refused to give his name.",
    ],
    "fix API rate limit errors": [
        "Our integration keeps getting HTTP 429 'too many requests' from your API since we scaled up.",
        "We hit your request quota within minutes and calls start failing. Can the limit be raised?",
    ],
    "fix an API key that is rejected": [
        "Every API call returns 401 'invalid API key' even though I just generated the key.",
        "Our backend can't authenticate to your API anymore, the token is reported as unauthorized.",
    ],
    "fix webhooks that are not delivered": [
        "We stopped receiving your webhook notifications for new orders, our endpoint is up and reachable.",
        "The order.created callbacks never arrive at our server anymore.",
    ],
    "ask for API documentation": [
        "Where is the reference documentation for your REST API endpoints?",
        "Do you have docs or an OpenAPI spec for the developer API?",
    ],
    "export a copy of my data": [
        "I'd like to download all my projects and files as an archive.",
        "Is there a way to export everything in my workspace to a zip file?",
    ],
    "ask where data is stored": [
        "For compliance we need to know which country your servers storing our data are located in.",
        "Is customer data hosted in the EU or in the US?",
    ],
    "stop marketing emails": [
        "Please stop sending me promotional newsletters, I only want receipts.",
        "How do I unsubscribe from your marketing mailing list?",
    ],
    "change the interface language": [
        "The app is showing everything in French. How do I switch it back to English?",
        "Can I set the dashboard language to Japanese?",
    ],
    "add a team member": [
        "I want to invite my new colleague to our workspace so she can edit documents.",
        "How do I give a new employee access to our team account?",
    ],
    "remove a team member": [
        "An employee left the company. Please take away his access to our workspace.",
        "How do I kick a former contractor out of our team account?",
    ],
    "transfer account ownership": [
        "I'm stepping down as admin. The account should belong to my coworker from now on.",
        "We need to hand over ownership of the organization account to our new CTO.",
    ],
    "request tax exemption": [
        "We're a registered non-profit. Can you stop charging us sales tax? I can send our certificate.",
        "Our school is tax-exempt, please remove VAT from future invoices.",
    ],
    "get a phone number for support": [
        "Is there a number I can call to talk to a human?",
        "I prefer speaking on the phone. What's your support hotline?",
    ],
    "book a product demo": [
        "We're evaluating tools for our 200-person company. Could someone walk us through the product in a live session?",
        "I'd like to schedule a demo call with your sales team next week.",
    ],
    "ask about a student discount": [
        "I'm a university student. Do you offer reduced pricing for students?",
        "Is there an academic discount if I verify with my .edu email?",
    ],
    "report a screen reader problem": [
        "I'm blind and use VoiceOver. The checkout buttons on your site aren't announced at all.",
        "Your forms can't be used with a screen reader, the fields have no labels.",
    ],
    "report a wrong translation": [
        "In the Spanish version, the 'Save' button is translated as 'Rescue'.",
        "The German text on the settings page is mistranslated and makes no sense.",
    ],
    "ask for a bulk order quote": [
        "We'd like to buy 500 units for our offices. Can you give us a volume price?",
        "What discount do you offer if we order 1,000 licenses at once?",
    ],
    "report a security vulnerability": [
        "I'm a security researcher and found an XSS flaw on your login page. How can I disclose it responsibly?",
        "Your password reset endpoint leaks user emails. I'd like to report this privately.",
    ],
    "recover a deleted file": [
        "I accidentally deleted an important spreadsheet yesterday. Can it be restored?",
        "Is there a trash or backup where I can get back a folder I removed by mistake?",
    ],
    "increase the storage limit": [
        "We're at 98% of our storage quota. How do we get more space?",
        "Can I buy extra storage without changing my whole plan?",
    ],
    "fix syncing between devices": [
        "Notes I write on my phone don't show up on my laptop anymore.",
        "Changes made on the desktop app never appear on my tablet.",
    ],
    "connect a device to wifi": [
        "The smart plug won't join my home network, the light just keeps blinking.",
        "How do I get the camera onto my wireless network? Setup fails at the network step.",
    ],
    "report a spammer or abusive user": [
        "A user keeps sending me threatening messages through your platform.",
        "Someone is posting scam links in every community thread. Please ban them.",
    ],
    "ask about business hours": [
        "What time does your support team open on Saturdays?",
        "Are you available on public holidays, and until what time on weekdays?",
    ],
}

# Requests that are easy to confuse share a group; their labels are offered first as distractors.
INTENT_GROUPS = {
    "account": ["reset a forgotten password", "change the account email address", "turn on two-factor authentication",
                "unlock an account locked after failed logins", "close and delete the account",
                "report a login by someone else", "transfer account ownership", "report a security vulnerability"],
    "billing": ["refund a duplicate charge", "update the saved payment card", "send a copy of an invoice",
                "change the billing address", "request tax exemption", "fix a promo code that is not accepted",
                "redeem a gift card", "return an item for a refund"],
    "plan": ["explain the price of each plan", "upgrade to a higher plan", "downgrade to a cheaper plan",
             "cancel the subscription", "pause the subscription for a while", "ask about a student discount",
             "ask for a bulk order quote", "book a product demo", "increase the storage limit"],
    "order": ["check where an order is", "complain about a late delivery", "change the delivery address of an order",
              "report an item damaged in shipping", "report receiving the wrong item", "return an item for a refund",
              "exchange an item for another size", "report an item missing from the package"],
    "product": ["claim warranty for a product that stopped working", "ask if a product is compatible",
                "get the user manual", "connect a device to wifi", "report an item damaged in shipping"],
    "tech": ["report the app crashing on start", "report the website being slow", "report the website being completely down",
             "report a broken link", "fix syncing between devices", "report a screen reader problem",
             "report a wrong translation", "change the interface language", "recover a deleted file"],
    "api": ["fix API rate limit errors", "fix an API key that is rejected", "fix webhooks that are not delivered",
            "ask for API documentation", "report the website being completely down"],
    "data": ["export a copy of my data", "ask where data is stored", "stop marketing emails", "close and delete the account",
             "report a security vulnerability", "recover a deleted file"],
    "team": ["add a team member", "remove a team member", "transfer account ownership"],
    "contact": ["suggest a new feature", "praise the product", "complain about a support agent",
                "get a phone number for support", "ask about business hours", "report a spammer or abusive user",
                "book a product demo"],
}

# Harder tickets: Japanese, and English ones that mention other topics or say what they are NOT asking.
INTENTS_JA = [
    ("パスワードを忘れてしまいました。再設定の方法を教えてください。", "reset a forgotten password"),
    ("登録しているメールアドレスを、会社のものから個人のものに変えたいです。", "change the account email address"),
    ("ログインのときに認証アプリのコードも求めるようにしたいのですが、どこで設定できますか？", "turn on two-factor authentication"),
    ("同じ注文で二回引き落とされています。一回分を返金してください。", "refund a duplicate charge"),
    ("経理に出すので、先月分の請求書をもう一度送ってもらえますか。", "send a copy of an invoice"),
    ("来月からはもう使わないので、月額プランを解約したいです。", "cancel the subscription"),
    ("三か月ほど海外出張なので、その間だけ契約を止めておけますか？データは残したいです。", "pause the subscription for a while"),
    ("月曜に注文した商品はもう発送されましたか？追跡番号が届いていません。", "check where an order is"),
    ("届いた箱がつぶれていて、中のグラスが割れていました。", "report an item damaged in shipping"),
    ("黒のスニーカーを注文したのに、白いサンダルが届きました。", "report receiving the wrong item"),
    ("シャツが小さかったので、Lサイズと交換してもらえますか。", "exchange an item for another size"),
    ("半年前に買った電気ケトルが急に電源が入らなくなりました。保証で修理できますか。",
     "claim warranty for a product that stopped working"),
    ("アップデートしてから、アプリを開くとすぐ落ちてしまいます。", "report the app crashing on start"),
    ("サイトにまったくつながりません。社内の誰もアクセスできない状態です。", "report the website being completely down"),
    ("APIを呼ぶと429エラーが頻発します。上限を引き上げてもらえませんか。", "fix API rate limit errors"),
    ("新規注文のWebhook通知がうちのサーバーに届かなくなりました。サーバー側は正常です。", "fix webhooks that are not delivered"),
    ("これまでのプロジェクトをすべてzipでダウンロードしたいです。", "export a copy of my data"),
    ("キャンペーンのお知らせメールはもう送らないでください。", "stop marketing emails"),
    ("画面が全部英語になってしまいました。日本語表示に戻すにはどうすればいいですか。", "change the interface language"),
    ("新しく入った同僚をワークスペースに招待したいです。", "add a team member"),
    ("退職した社員がまだワークスペースに入れるので、権限を外してください。", "remove a team member"),
    ("電話で直接相談したいのですが、サポートの電話番号はありますか。", "get a phone number for support"),
    ("大学生なのですが、学割はありますか？", "ask about a student discount"),
    ("昨日うっかり大事な資料を削除してしまいました。元に戻せますか。", "recover a deleted file"),
    ("容量がほぼいっぱいです。プランはそのままで容量だけ増やせますか。", "increase the storage limit"),
    ("スマホで書いたメモがパソコンに反映されません。", "fix syncing between devices"),
    ("コミュニティで詐欺サイトのリンクを貼りまくっている人がいます。対処してください。", "report a spammer or abusive user"),
    ("セキュリティ研究者です。ログイン画面に脆弱性を見つけたので、非公開で報告したいです。", "report a security vulnerability"),
    ("パスワードを何度か間違えたらアカウントがロックされました。今日中に使いたいので解除してください。",
     "unlock an account locked after failed logins"),
    ("到着予定日を1週間過ぎても荷物が届きません。いい加減にしてほしいです。", "complain about a late delivery"),
]
INTENTS_TRICKY = [
    ("Loving the app so far! Quick thing though: my teammate Jen just joined and I can't figure out how to get her "
     "into our workspace.", "add a team member"),
    ("Not complaining about the price at all, but we've stopped using half the features, so a smaller plan would make "
     "more sense for us.", "downgrade to a cheaper plan"),
    ("The password is fine, I know it. It just says I've been locked out for too many tries.",
     "unlock an account locked after failed logins"),
    ("Order #4410 arrived on time, thanks. Unfortunately it's a size too big. Can I get the next size down instead?",
     "exchange an item for another size"),
    ("I'm not asking for a refund, I just need a copy of the July invoice for our records.", "send a copy of an invoice"),
    ("Your webhooks work great. Your API docs, on the other hand: where are they? I can't find the endpoint reference "
     "anywhere.", "ask for API documentation"),
    ("We are fine with being charged, but we're a registered charity and shouldn't be paying sales tax on it.",
     "request tax exemption"),
    ("Before deciding whether to renew, I'd like to compare what each tier would cost us per month.",
     "explain the price of each plan"),
    ("It's not that the site is down. It loads, it just takes forever on every click.", "report the website being slow"),
    ("My phone and laptop show different versions of the same document; edits from one never reach the other.",
     "fix syncing between devices"),
    ("Someone in your support chat was incredibly dismissive yesterday. I want this escalated to a manager.",
     "complain about a support agent"),
    ("I don't want to cancel, just stop billing me for June and July while I'm away, then pick up again in August.",
     "pause the subscription for a while"),
    ("The app doesn't crash, but every time I open it, it's in Portuguese, which I don't speak.",
     "change the interface language"),
    ("We have 300 laptops to equip with your software. What price can you do for that many seats?",
     "ask for a bulk order quote"),
    ("This isn't a feature request: your checkout page cannot be operated with a screen reader at all, and our blind "
     "employees are stuck.", "report a screen reader problem"),
    ("Got a text saying my account was accessed from a new device in Ukraine. I've never been there.",
     "report a login by someone else"),
    ("Tracking says delivered, but the box only had the camera. The lens that was part of the bundle is not there.",
     "report an item missing from the package"),
    ("Your newsletter is nice, but I really only want receipts in my inbox, nothing else.", "stop marketing emails"),
    ("Not sure who to ask: our auditor needs to know whether our files sit on servers in Germany or somewhere else.",
     "ask where data is stored"),
    ("Can you hand the admin role and the account itself over to Priya? I'm leaving the company next week.",
     "transfer account ownership"),
    ("Our integration started failing with 'unauthorized' right after we rotated keys, even though the new key is "
     "copied correctly.", "fix an API key that is rejected"),
    ("The speaker works fine over Bluetooth, but I can't get it onto the home Wi-Fi network.", "connect a device to wifi"),
    ("My card number changed after the bank replaced it. Please charge the new one from now on.",
     "update the saved payment card"),
    ("I'd love to see what your product does before we commit. Could your team show it to us on a call?",
     "book a product demo"),
    ("I wasn't charged twice, but the total I paid is wrong because SAVE10 was ignored at checkout.",
     "fix a promo code that is not accepted"),
    ("The headphones are not broken, I just don't like them. How do I send them back and get my money?",
     "return an item for a refund"),
    ("I moved to Osaka last month. The parcel I ordered an hour ago still shows my Tokyo address. Can it go to the "
     "new place?", "change the delivery address of an order"),
    ("The French page translates 'Sign out' as 'Sign up', which confused our customers.", "report a wrong translation"),
    ("Everything about your service is fantastic, and I mean it. Thank you, team!", "praise the product"),
    ("Not a bug report exactly: it would be really handy if exports could also be in CSV, not just PDF.",
     "suggest a new feature"),
]

COLORS = {"red": (220, 30, 30), "blue": (30, 70, 220), "green": (30, 160, 50), "yellow": (240, 200, 0),
          "purple": (130, 40, 170), "orange": (250, 130, 0), "black": (0, 0, 0), "gray": (140, 140, 140)}
SHAPES = ["circle", "square", "triangle", "star", "hexagon", "cross"]

WORDS = ["APPLE", "BRIDGE", "CANDLE", "DOCTOR", "EAGLE", "FOREST", "GARDEN", "HAMMER", "ISLAND", "JACKET",
         "KITCHEN", "LADDER", "MARKET", "NEEDLE", "ORANGE", "PENCIL", "QUEEN", "RABBIT", "SILVER", "TICKET",
         "UMBRELLA", "VALLEY", "WINDOW", "YELLOW", "ZEBRA", "ANCHOR", "BASKET", "CASTLE", "DESERT", "ENGINE",
         "FLOWER", "GUITAR", "HARBOR", "INSECT", "JUNGLE", "KETTLE", "LEMON", "MIRROR", "NAPKIN", "OCEAN",
         "PILLOW", "RIVER", "SADDLE", "TABLE", "TUNNEL", "VIOLIN", "WALLET", "BUTTON", "CARPET", "DRAGON",
         "FINGER", "GOLDEN", "HELMET", "LIZARD", "MONKEY", "PARROT", "ROCKET", "SPIDER", "TOMATO", "WINTER"]

# label -> (dialog title, dialog message)
ERRORS = {
    "the payment card was declined": ("Payment failed", "Your card was declined by the issuing bank (code 51)."),
    "the login session expired": ("Signed out", "Your session has expired. Please sign in again."),
    "the device storage is full": ("Storage full", "There is not enough free space on this device to continue."),
    "there is no internet connection": ("Offline", "No network connection. Check your Wi-Fi or mobile data."),
    "the uploaded file is too large": ("Upload failed", "The file exceeds the maximum size of 25 MB."),
    "the password is wrong": ("Sign-in error", "The password you entered is incorrect."),
    "the server had an internal error": ("Error 500", "Something went wrong on our servers. Try again later."),
    "the page was not found": ("404", "The page you are looking for does not exist."),
    "the account is suspended": ("Account suspended", "This account has been suspended for violating the terms."),
    "the app must be updated": ("Update required", "This version is no longer supported. Install the latest version."),
    "access to the folder is denied": ("Access denied", "You don't have permission to open this folder."),
    "the file format is not supported": ("Cannot open file", "The .xyz format is not supported."),
    "the software license has expired": ("License expired", "Your license expired on 03/01. Renew to keep using it."),
    "camera access is blocked": ("Camera unavailable", "Allow camera access in Settings to take photos."),
    "the printer is offline": ("Printer offline", "The selected printer is not responding."),
    "the battery is almost empty": ("Low battery", "5% battery remaining. Connect a charger."),
    "the email address is already registered": ("Sign-up failed", "An account with this email already exists."),
    "the verification code is wrong": ("Verification failed", "The 6-digit code you entered is not valid."),
    "the request timed out": ("Timeout", "The server took too long to respond."),
    "the coupon has expired": ("Coupon not applied", "This promotion ended on 12/31."),
    "the item is out of stock": ("Unavailable", "Sorry, this item is currently out of stock."),
    "the address could not be verified": ("Invalid address", "We couldn't verify the shipping address you entered."),
    "too many login attempts were made": ("Locked", "Too many attempts. Try again in 30 minutes."),
    "the browser is not supported": ("Unsupported browser", "Please use a recent version of Chrome, Firefox or Safari."),
    "the download was blocked as unsafe": ("Download blocked", "This file may harm your computer and was blocked."),
    "the meeting link is invalid": ("Cannot join", "This meeting ID does not exist or has ended."),
    "the database connection failed": ("Database error", "Could not connect to the database server."),
    "the bluetooth device failed to pair": ("Pairing failed", "Couldn't connect to 'Speaker-01'. Make sure it is in pairing mode."),
    "the backup failed": ("Backup incomplete", "The last backup could not be completed."),
    "the GPS signal is lost": ("Location unavailable", "Searching for GPS signal..."),
}

TASKS = {
    "intent": {"instructions": "Which request best describes this support ticket?", "pool": list(INTENTS)},
    "shape": {"instructions": "Which object is shown in the image?",
              "pool": [f"a {c} {s}" for c in COLORS for s in SHAPES]},
    "count": {"instructions": "How many black dots are in the image?",
              "pool": ["1 dot"] + [f"{n} dots" for n in range(2, 31)]},
    "word": {"instructions": "Which word is written on the sign in the image?", "pool": WORDS},
    "error": {"instructions": "Which problem does the user's screenshot show?", "pool": list(ERRORS)},
    # the corner is named per item (item["instructions"]); the other three objects are the first distractors
    "scene": {"instructions": "Which object is in the given corner of the image?",
              "pool": [f"a {c} {s}" for c in COLORS for s in SHAPES]},
    # the other amounts printed on the receipt are the first distractors
    "receipt": {"instructions": "What is the total amount on the receipt?",
                "pool": [f"${random.Random(i).randint(300, 9999) / 100:.2f}" for i in range(60)]},
}
IMAGE_TEXT = {
    "shape": "Here is a picture.",
    "count": "Here is a picture with some dots.",
    "word": "Here is a photo of a sign.",
    "error": "This message keeps popping up on my screen. What is wrong?",
    "scene": "Here is a picture with four objects.",
    "receipt": "Here is my receipt from today.",
}
CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right"]
SHOP_ITEMS = ["Coffee beans", "Milk 1L", "Bread", "Eggs x10", "Bananas", "Cheese", "Pasta", "Tomato sauce", "Rice 2kg",
              "Yogurt", "Apples", "Butter", "Orange juice", "Chicken", "Chocolate", "Tea bags", "Cereal", "Soap"]
MASTER = 1024  # images are stored at this size; the benchmark resizes them to the requested size


def _font(size, bold=True):
    from PIL import ImageFont
    for name in (["DejaVuSans-Bold.ttf"] if bold else []) + ["DejaVuSans.ttf"]:
        for d in ["/usr/share/fonts/truetype/dejavu/", ""]:
            try:
                return ImageFont.truetype(d + name, size)
            except OSError:
                pass
    return ImageFont.load_default(size)


def _poly(cx, cy, r, n, rot=-90, inner=None):
    import math
    pts = []
    for i in range(n * (2 if inner else 1)):
        rr = (inner if inner and i % 2 else r)
        a = math.radians(rot + i * 360 / (n * (2 if inner else 1)))
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def draw_shape(color, shape, rng, img=None, r=None, cx=None, cy=None):
    from PIL import Image, ImageDraw
    if img is None:
        img = Image.new("RGB", (MASTER, MASTER), "white")
        r = rng.randint(150, 230); cx = rng.randint(r + 40, MASTER - r - 40); cy = rng.randint(r + 40, MASTER - r - 40)
    d = ImageDraw.Draw(img)
    c = COLORS[color]
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=c)
    elif shape == "triangle":
        d.polygon(_poly(cx, cy, r, 3), fill=c)
    elif shape == "star":
        d.polygon(_poly(cx, cy, r, 5, inner=r * 0.42), fill=c)
    elif shape == "hexagon":
        d.polygon(_poly(cx, cy, r, 6, rot=0), fill=c)
    elif shape == "cross":
        w = r // 3
        d.rectangle([cx - r, cy - w, cx + r, cy + w], fill=c); d.rectangle([cx - w, cy - r, cx + w, cy + r], fill=c)
    return img


def draw_dots(n, rng):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (MASTER, MASTER), "white"); d = ImageDraw.Draw(img)
    r, pts = 34, []
    while len(pts) < n:
        p = (rng.randint(r + 20, MASTER - r - 20), rng.randint(r + 20, MASTER - r - 20))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > (3 * r) ** 2 for q in pts):
            pts.append(p)
    for x, y in pts:
        d.ellipse([x - r, y - r, x + r, y + r], fill="black")
    return img


def draw_sign(word, rng):
    from PIL import Image, ImageDraw
    bg = rng.choice([(70, 120, 70), (60, 90, 150), (150, 60, 60), (90, 90, 90)])
    img = Image.new("RGB", (MASTER, MASTER), (200, 220, 235)); d = ImageDraw.Draw(img)
    d.rectangle([490, 560, 534, 1024], fill=(110, 110, 110))  # post
    f = _font(130); w = d.textlength(word, font=f)
    x0 = (MASTER - w) / 2 - 50; y0 = rng.randint(220, 360)
    d.rounded_rectangle([x0, y0, x0 + w + 100, y0 + 230], radius=30, fill=bg, outline="white", width=10)
    d.text(((MASTER - w) / 2, y0 + 45), word, font=f, fill="white")
    return img


def draw_error(title, msg, rng):
    from PIL import Image, ImageDraw
    import textwrap
    img = Image.new("RGB", (MASTER, MASTER), rng.choice([(235, 238, 242), (40, 44, 52), (210, 225, 245)]))
    d = ImageDraw.Draw(img)
    for i in range(6):  # blurred-looking app content behind the dialog
        y = 80 + i * 150
        d.rounded_rectangle([60, y, 964, y + 110], radius=16, fill=(200, 205, 212))
    x0, y0, x1, y1 = 140, 300, 884, 720
    d.rounded_rectangle([x0 + 8, y0 + 10, x1 + 8, y1 + 10], radius=24, fill=(120, 120, 120))
    d.rounded_rectangle([x0, y0, x1, y1], radius=24, fill="white", outline=(180, 180, 180), width=3)
    d.ellipse([x0 + 40, y0 + 40, x0 + 120, y0 + 120], fill=(220, 50, 50))
    d.text((x0 + 80, y0 + 80), "!", font=_font(60), fill="white", anchor="mm")
    d.text((x0 + 150, y0 + 52), title, font=_font(52), fill="black")
    for i, line in enumerate(textwrap.wrap(msg, 30)):
        d.text((x0 + 50, y0 + 160 + i * 52), line, font=_font(40, bold=False), fill=(50, 50, 50))
    d.rounded_rectangle([x1 - 200, y1 - 100, x1 - 40, y1 - 30], radius=14, fill=(40, 110, 230))
    d.text((x1 - 120, y1 - 65), "OK", font=_font(36), fill="white", anchor="mm")
    return img


def draw_scene(objs, rng):
    """objs: 4 (color, shape) pairs, drawn in the corners in CORNERS order."""
    from PIL import Image
    img = Image.new("RGB", (MASTER, MASTER), "white")
    for i, (c, s) in enumerate(objs):
        cx = 256 + 512 * (i % 2) + rng.randint(-40, 40); cy = 256 + 512 * (i // 2) + rng.randint(-40, 40)
        draw_shape(c, s, rng, img=img, r=rng.randint(120, 170), cx=cx, cy=cy)
    return img


def draw_receipt(lines, subtotal, tax, total, cash, change, rng):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (MASTER, MASTER), (120, 110, 100)); d = ImageDraw.Draw(img)
    x0 = rng.randint(200, 260); x1 = x0 + 560
    d.rectangle([x0, 40, x1, 984], fill=(250, 250, 245))
    f, fb = _font(34, bold=False), _font(38)
    d.text(((x0 + x1) / 2, 100), rng.choice(["GREEN MART", "CITY GROCER", "DAILY FOODS", "CORNER SHOP"]),
           font=_font(46), fill="black", anchor="mm")
    d.text(((x0 + x1) / 2, 150), f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}  "
           f"{rng.randint(8, 21)}:{rng.randint(0, 59):02d}", font=f, fill=(60, 60, 60), anchor="mm")
    y = 220
    for name, price in lines:
        d.text((x0 + 30, y), name, font=f, fill="black"); d.text((x1 - 30, y), price, font=f, fill="black", anchor="ra")
        y += 52
    d.line([x0 + 30, y + 10, x1 - 30, y + 10], fill="black", width=2); y += 30
    for name, price, font in [("Subtotal", subtotal, f), ("Tax 8%", tax, f), ("TOTAL", total, fb),
                              ("Cash", cash, f), ("Change", change, f)]:
        d.text((x0 + 30, y), name, font=font, fill="black"); d.text((x1 - 30, y), price, font=font, fill="black", anchor="ra")
        y += 56
    d.text(((x0 + x1) / 2, y + 60), "Thank you!", font=f, fill=(60, 60, 60), anchor="mm")
    return img


def build(out_dir):
    out = pathlib.Path(out_dir); (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random("eval_v1")
    items = []

    def near(label):  # labels in the same group(s) as this request
        return sorted({l for g in INTENT_GROUPS.values() if label in g for l in g} - {label})

    for li, (label, texts) in enumerate(INTENTS.items()):
        for ti, text in enumerate(texts):
            items.append({"id": f"intent-{li:02d}{ti}", "task": "intent", "text": text, "image": None, "gold": label,
                          "near": near(label)})
    for name, rows in [("ja", INTENTS_JA), ("tricky", INTENTS_TRICKY)]:
        for i, (text, label) in enumerate(rows):
            assert label in INTENTS, label
            items.append({"id": f"intent-{name}-{i:02d}", "task": "intent", "text": text, "image": None,
                          "gold": label, "near": near(label)})
    combos = [(c, s) for c in COLORS for s in SHAPES]
    for i, (c, s) in enumerate(rng.sample(combos, 30)):
        p = f"images/shape-{i:02d}.png"; draw_shape(c, s, rng).save(out / p)
        items.append({"id": f"shape-{i:02d}", "task": "shape", "text": IMAGE_TEXT["shape"], "image": p,
                      "gold": f"a {c} {s}"})
    counts = list(range(1, 16)) + rng.sample(range(1, 16), 5)
    for i, n in enumerate(counts):
        p = f"images/count-{i:02d}.png"; draw_dots(n, rng).save(out / p)
        items.append({"id": f"count-{i:02d}", "task": "count", "text": IMAGE_TEXT["count"], "image": p,
                      "gold": "1 dot" if n == 1 else f"{n} dots"})
    for i, w in enumerate(rng.sample(WORDS, 30)):
        p = f"images/word-{i:02d}.png"; draw_sign(w, rng).save(out / p)
        items.append({"id": f"word-{i:02d}", "task": "word", "text": IMAGE_TEXT["word"], "image": p, "gold": w})
    for i, lab in enumerate(rng.sample(list(ERRORS), 20)):
        p = f"images/error-{i:02d}.png"; draw_error(*ERRORS[lab], rng).save(out / p)
        items.append({"id": f"error-{i:02d}", "task": "error", "text": IMAGE_TEXT["error"], "image": p, "gold": lab})
    for i in range(30):
        objs = rng.sample(combos, 4); corner = i % 4
        p = f"images/scene-{i:02d}.png"; draw_scene(objs, rng).save(out / p)
        labels = [f"a {c} {s}" for c, s in objs]
        items.append({"id": f"scene-{i:02d}", "task": "scene", "text": IMAGE_TEXT["scene"], "image": p,
                      "instructions": f"Which object is in the {CORNERS[corner]} corner of the image?",
                      "gold": labels[corner], "near": [l for l in labels if l != labels[corner]]})
    for i in range(20):
        lines = [(n, f"${rng.randint(100, 2500) / 100:.2f}") for n in rng.sample(SHOP_ITEMS, rng.randint(3, 6))]
        sub = sum(float(pr[1:]) for _, pr in lines); tax = round(sub * 0.08, 2)
        tot = round(sub + tax, 2); cash = (int(tot) // 50 + 1) * 50  # paid in cash: a larger amount than the total
        s_sub, s_tax, s_tot, s_cash, s_chg = (f"${x:.2f}" for x in (sub, tax, tot, cash, cash - tot))
        p = f"images/receipt-{i:02d}.png"; draw_receipt(lines, s_sub, s_tax, s_tot, s_cash, s_chg, rng).save(out / p)
        others = sorted(({pr for _, pr in lines} | {s_sub, s_tax, s_cash, s_chg}) - {s_tot})
        items.append({"id": f"receipt-{i:02d}", "task": "receipt", "text": IMAGE_TEXT["receipt"], "image": p,
                      "gold": s_tot, "near": others})
    (out / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    (out / "tasks.json").write_text(json.dumps(TASKS, indent=1, ensure_ascii=False))
    print(f"{len(items)} items -> {out}")


def load_items(eval_dir):
    d = pathlib.Path(eval_dir)
    return json.loads((d / "items.json").read_text()), json.loads((d / "tasks.json").read_text())


def make_question(item, tasks, k, seed=0):
    """-> (record in /v1/systemone form without images, gold option id). Options are o000..o{k-1}."""
    t = tasks[item["task"]]
    rng = random.Random(f"{seed}-{item['id']}-{k}")
    near = [l for l in item.get("near", []) if l != item["gold"]]
    rng.shuffle(near)
    near = near[:k - 1]  # confusable labels first ...
    pool = [l for l in t["pool"] if l != item["gold"] and l not in near]
    assert k - 1 - len(near) <= len(pool), f"K={k} is too large for the {item['task']} task"
    opts = near + rng.sample(pool, k - 1 - len(near)) + [item["gold"]]  # ... then random ones from the pool
    rng.shuffle(opts)
    criteria = {f"o{j:03d}": lab for j, lab in enumerate(opts)}
    gold_id = next(i for i, lab in criteria.items() if lab == item["gold"])
    key = "ticket" if item["task"] == "intent" else "message"
    rec = {"state": {key: {"id": item["id"], "body": item["text"]}},
           "questions": {"route": {"type": "choice", "instructions": item.get("instructions", t["instructions"]),
                                   "criteria": criteria}}}
    return rec, gold_id


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "dataset/eval_v1")
