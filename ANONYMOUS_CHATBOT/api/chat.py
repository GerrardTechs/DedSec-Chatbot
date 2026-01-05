from http.server import BaseHTTPRequestHandler
import json
import pickle
import os

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        # Load model
        with open('chatbot_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Process request
        response = chatbot_response(data['message'])
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())