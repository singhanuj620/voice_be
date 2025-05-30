# Voice Assistant Report Analyzer

### Codebase setup

- Create .env file in root folder and add
  - ` GOOGLE_API_KEY=`
- Create Gemini LLM key at [Gemini Studio](https://aistudio.google.com/)
- Make a folder name 'secrets' at root level, and create a new json file named 'google_tts_key'
  - Place the google tts config inside it
  - In tts.py file, uncomment the L4 and edit the actual path of this google_tts_key file path
  - Comment out L7 i.e /etc/secrets/google_tts_key.json line
  - Do the same with 
