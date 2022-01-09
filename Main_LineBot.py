from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
import main as ma
app = Flask(__name__)
# LINE BOT info
line_bot_api = LineBotApi('ZwmVJ7cfE3BMvhjatt74oBrqvfS4YQ8xA7yHVwCjsugTpCKwP60+kPpe8w6Oq9V61nNY6WIZ7xO4/EfUupLexv3Dj/egs5BMUs71COk8KenxbgD9VWaRY/5yYcCx3uz8tiUN7JXhorZ0f7GYu/dS4AdB04t89/1O/w1cDnyilFU=')   #fill in your "Channel Access token"
handler = WebhookHandler('8fbd6804de73ffb593a45392a44451f6')          #fill in your "Channel Secert"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print(body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# Message event
@handler.add(MessageEvent)
def handle_message(event):
    message_type = event.message.type
    user_id = event.source.user_id
    reply_token = event.reply_token
    #message = event.message.text #input message
    message = []
    tmp=""
    if event.message.text == "start":
        #print('start')
        #message.append('start the program')
        #line_bot_api.reply_message(reply_token, TextSendMessage(text = "start the program"))
        
        mess = ma.program()
        tmp += f"temperature : {mess[0]}℃ humidity : {mess[1]}% emotion : {mess[2]} result : {mess[3]}"
        print(type(mess[0]))
        line_bot_api.reply_message(reply_token, TextSendMessage(text = tmp))
        #line_bot_api.reply_message(reply_token, TextSendMessage(text = temp))
    else:
        line_bot_api.reply_message(reply_token, TextSendMessage(text = "frank"))
    #line_bot_api.reply_message(reply_token, TextSendMessage(text = "frank"))
import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)

