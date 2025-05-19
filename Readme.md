# Voice Assistant Report Analyzer

### Codebase setup

- Create .env file in root folder and add
  - ` LLM_API_KEY=`
- Create Gemini LLM key at [Gemini Studio](https://aistudio.google.com/)
- Install ffmpeg in windows
  - [Download windows exe build](https://www.gyan.dev/ffmpeg/builds/)
  - Make a folder name 'ffmpeg' in C:/Program Files
  - Extract this build inside that
  - Copy the path
  - Win + S -> Edit Enviornment variables -> Path -> Add -> Paste it -> Save
  - Restart VS Code
  - To check if ffmpeg is installed correctly, run ffmpeg -version
