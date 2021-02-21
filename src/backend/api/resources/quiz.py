# logic for /artifacts
from src.backend.util.google_speech_to_text import get_text_corpus_from_audio
from src.backend.util.qa_gen import get_qa_pairs_from_text_corpus
from flask import abort, jsonify, request, make_response, Blueprint, url_for, Response
from flask_restful import reqparse, Resource, fields, marshal
import json


class GenerateQuizAPI(Resource):
    def __init__(self):
        super(GenerateQuizAPI, self).__init__()

    def post(self):
        params = request.json
        url = params['url'] if 'url' in params else None
        text = params['text'] if 'text' in params else None

        if not url and not text:
            abort(400, description="invalid params")
        if text and len(text) < 5:
            abort(400, description="insufficient text corpus length")
        
        if url:
            text = get_text_corpus_from_audio(url)

        qa_pairs = get_qa_pairs_from_text_corpus(text)
        response = jsonify(qa_pairs)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.status_code = 200
        return response
            
