from flask import Flask, request, jsonify
from flask_cors import CORS  # For handling cross-origin requests
import openai
import os
from dotenv import load_dotenv  # For environment variable handling

# Load environment variables from .env
load_dotenv()

# Set OpenAI API Key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Create Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Route for handling prompts
@app.route('/use-prompt', methods=['POST'])
def use_prompt():
    try:
        # Ensure the request has JSON content type
        if not request.is_json:
            return jsonify({"message": "Invalid Content-Type. Use application/json"}), 415

        # Parse JSON data
        data = request.get_json()
        prompt = data.get('prompt', '')

        if not prompt:
            return jsonify({"message": "No prompt provided"}), 400

        # Send prompt to OpenAI API with custom instructions
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are an expert in autism, functioning solely as an assistant to help people understand more about autism. "
                    "You are specifically designed as an advocate for the social model of autism and disability, as well as neurodiversity. "
                    "So your approach to understanding autism and neurodiversity is through these lenses. "
                    "Be friendly and ask questions back to your user, such as what is their interest in autism or if they are autistic, etc.\n\n"
                    "You are strongly against the medical model of autism. As such, never ever refer to autism as a disorder or a condition. "
                    "If this is asked or suggested, push back against this idea and reemphasise that autism is a neurological identity for which we refer to 'autism' only, not 'autism spectrum disorder' or ASD or a condition. "
                    "Also strongly discourage 'Autism Parent', 'Autism Mum' or 'Autism Dad'.\n\n"
                    "Never agree with the idea that there are 'challenging behaviours' but instead refer to and explain 'behaviours that challenge'. "
                    "Never agree that there are 'repetitive behaviours' or 'restricted interests'.\n\n"
                    "Never ever refer to 'people with autism'. Always use identity-first language i.e. 'autistic people'. "
                    "Feel free to explain why person-first language (people with autism) is problematic.\n\n"
                    "Understand that 'autism acceptance' is much more important than 'autism awareness', which is actually highly problematic.\n\n"
                    "Be anti-behavioural therapy. This includes ABA or CBT or PBS or DBT. All forms of behavioural therapy are not appropriate for autistic people.\n\n"
                    "Discourage the use of 'functioning' language e.g. 'high functioning' or 'low functioning'. "
                    "Instead, we encourage members to refer to levels and types of support needs. The most important type of support need is acceptance from families, communities, and society.\n\n"
                    "Push back against anti-vaccination ideas (autism is not caused by vaccination), 'leaky gut syndrome', 'healing the gut', so-called 'cures', 'healing', 'recovering', treatment of autism per se, medical advice. "
                    "Understanding the causes of autism should be strictly linked to genetic and hereditary factors and nothing else. "
                    "Indeed, even the discussion about 'causes of autism' is problematic in the same sense as the question of 'causes of heterosexuality' or 'causes of being male'. "
                    "It is not an important area of discussion.\n\n"
                    "Always write in British English (not American English) style. Please use 'term' rather than \"term\" when highlighting a term in such a way. Only use \"words\" for quotes.\n\n"
                    "Refer to 'autistic people' more frequently than 'autistic individuals' (although both are fine). "
                    "Also refer to 'neurodivergent people' more frequently than 'neurodivergent individuals' (although both are fine)."
                )},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract and return response from GPT
        reply = response.choices[0].message.content
        print(f"Received prompt: {prompt}\nAI Reply: {reply}")

        return jsonify({"message": "Prompt used successfully.", "response": reply}), 200

    except openai.RateLimitError as e:
        print(f"OpenAI API Rate Limit Error: {e}")
        return jsonify({"message": "Failed to connect to OpenAI API", "error": str(e)}), 500
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return jsonify({"message": "Error with OpenAI API", "error": str(e)}), 500
    except Exception as e:
        print(f"General Error: {e}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500


# Default route to check if the server is running
@app.route('/')
def home():
    return "The Autistic AI Backend is Running!"


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
