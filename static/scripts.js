function toggleDropdown() {
    const content = document.getElementById('dropdownContent');
    content.style.display = content.style.display === 'none' ? 'block' : 'none';
}

function showLoadingSpinner() {
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('processingMessage').style.display = 'block';  // Show the message
}


function displayTextOneWordAtATime(text, elementId) {
    const element = document.getElementById(elementId);
    element.innerHTML = '';
    const words = text.split(' ');
    let index = 0;

    const interval = setInterval(() => {
        if (index < words.length) {
            element.innerHTML += words[index] + ' ';
            index++;
        } else {
            clearInterval(interval);
        }
    }, 100);  // Adjust the interval time to control the speed of the text rendering
}

document.addEventListener('DOMContentLoaded', () => {
    const forArgument = document.getElementById('forArgument');
    const againstArgument = document.getElementById('againstArgument');

    if (forArgument) {
        displayTextOneWordAtATime(forArgument.textContent.trim(), 'forArgument');
    }

    if (againstArgument) {
        displayTextOneWordAtATime(againstArgument.textContent.trim(), 'againstArgument');
    }
});

function continueDebate() {
    const previousStatement = document.getElementById('previous_statement').value;
    const userResponse = document.getElementById('user_response').value;

    fetch('/continue_debate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            previous_statement: previousStatement,
            user_response: userResponse
        }),
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('loadingSpinner').style.display = 'none';
            if (data.counter_argument) {
                displayTextOneWordAtATime(data.counter_argument, 'counter-argument');
            }
        })
        .catch((error) => {
            console.error('Error:', error);
        });
}

function submitFeedback(event) {
    event.preventDefault(); // Prevent the form from submitting the traditional way

    const form = document.getElementById('feedbackForm');
    const formData = new FormData(form);

    fetch('/feedback', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            // Display a message to the user
            const feedbackMessage = document.getElementById('feedbackMessage');
            feedbackMessage.innerText = data.message;
            feedbackMessage.style.display = 'block';

            // Optionally reset the form or disable it after submission
            form.reset();
        })
        .catch(error => {
            console.error('Error submitting feedback:', error);
        });
}


function continueDebate() {
    const userResponse = document.getElementById('user_response').value;
    const previousStatement = document.getElementById('previous_statement').value;

    fetch('/continue_debate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            previous_statement: previousStatement,
            user_response: userResponse
        })
    })
        .then(response => response.json())
        .then(data => {
            const counterArgumentDiv = document.getElementById('counter-argument');
            counterArgumentDiv.innerText = data.counter_argument;
        })
        .catch(error => {
            console.error('Error continuing debate:', error);
        });
}
