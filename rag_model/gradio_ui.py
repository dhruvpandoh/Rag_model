import gradio as gr
import requests

# Send question to FastAPI endpoint and get response
def get_rag_response(question):
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question}
        )

        # Raise exception for bad status codes
        response.raise_for_status()
        return response.json()['answer']
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to the server. Make sure the FastAPI server is running.\nDetails: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=get_rag_response,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Ask your ROS2 question here...",
        label="Question"
    ),
    outputs=gr.Textbox(
        lines=10, 
        label="Answer"
    ),
    title="ROS2 Documentation Assistant",
    description="Ask questions about ROS2 navigation, planning, and more!",
    examples=[
        ["How can I navigate to a specific pose in ROS2? Include replanning details."],
        ["What are the key components of the Nav2 stack?"],
        ["How do I implement recovery behaviors in navigation?"]
    ],
    theme=gr.themes.Soft()
)

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False
    )

