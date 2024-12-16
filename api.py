
import requests
import json

def get_access_token(client_id,client_secret):
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    client_id = client_id
    client_secret = client_secret
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

def main(client_id,client_secret):
    client_id = client_id
    client_secret = client_secret
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token(client_id,client_secret)
    
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "介绍一下北京"
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.text)
    

if __name__ == '__main__':
    client_id = 'YvRpjeGR9eCPQpH3RywalJnA'
    client_secret = '3CnLlEzp59hwDlyne4JBnyW5EU4rtHlZ'
    main(client_id,client_secret)