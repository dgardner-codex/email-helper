# email-helper
Categorize emails

Problem Statement

As context, in my email client there are folders to which inbox emails can be moved and stored.  When I ask the email client to store one or more of these inbox emails, it will make a suggestion as to which folder the email, or group of emails, should go.  

The problem is that the “suggestion” algorithm seems to only look for an appropriate email folder suggestion based on a keyword search in the “sender” and/or the “subject” of the email.  For instance, when the email client is asked to categorize an email where the sender is “ADT Security Services”  it will suggest the folder “ADT.”  Using only a keyword search however can lead to miscategorized  emails, or failure to make a suggestion at all.  Furthermore, the email client doesn’t identify emails that need a reply, or are likely junk emails.  There is no semantic intelligence in this process.

Solution Description

The desired outcome is that emails can be accurately labeled according to the available email folders, including “junk.”  In addition, I would like to see a priority of “high” associated with those emails that need a response regardless of the email folder suggested.   Non-response email can be labeled with a “normal” priority.  Junk email does not need a response, its priority will always be “normal.”  We may need to make a call to an agent to determine if the email is junk.  I’m not sure how difficult that determination with high confident is.  If an email doesn’t fit in a category and it isn’t junk, then put it in the “archive” email folder.  Archived email can need a response, or high priority.  Every email provided as input to be solved, will be solved.

I would suggest that this is a classification problem and can be solved using classification techniques often found in RAG systems.  

You have at your disposal, (1) a file containing a complete JSON list of email categories, (2) A file of sample emails in JSON format with the category labels and the reply label solved.  Use these for supporting your own generation of the correct labeling. We’ll hard code the filenames for now.  The input file to be solved, that is passed on the command line, will have this same format.

categories.json
{
	“Inbox”,
	“Drafts”,
	“Sent”,
	“Junk”,
	“Trash”,
	“Archive”,
	“2024 Nantucket Trip”,
	“ADT”,
	“Amazon”,
	“Apple”,
	“Boost”,
	“Cube Smart”,
	“CVS”,
	“DeepLearning”,
	“GoDaddy”,
	“Microsoft”,
	“Netflix”,
	“Phyllis”,
	“TLDR”,
	“Venmo”,
	“WK”
}

samples.json
[
	{
  	“date”: “mm/dd/yyyy”,
	“from": "sender@example.com",
	"subject": “email subject",
	“priority”: “high”,
	“category”: “text”,
 	"body": “text"
	},

	{
  	“date”: “mm/dd/yyyy”,
	“from": "sender@example.com",
	 "subject": “email subject",
	 “priority”: “normal”,
	 “category”: “text”,
 	 "body": “text”
	},
…
]

	
Input

You will be given on the command line a filename which contains a dictionary of emails that need the labels completed. This file will be in the same format as the sample file mentioned above.

Output

Write a new JSON file with an appropriate name that contains the input emails with the labels and response status completed.  Use the same JSON format as the input file.

Tracing

I also need a file that contains sufficient human-readable statements so I can understand what our code is doing and for debugging.  This doesn’t need a boolean switch.  Keep it simple. Something like:

def _trace(msg: str) -> None:
	"""
	Append messages to trace.txt.
	trace.txt is not cleared.
	"""
	with open("trace.txt","a") as f:
			f.write(msg + "\n")

Console msgs should be created using the print() function to indicate that the program is progressing.

Constraints:

Your role is a ChatGPT 5.3 Codex programmer and you can expect my feedback.
We are programming in Python.  The Python version is 3.12. No backwards compatibility needed.
The Python code should be “functional” vs object-oriented.
We can assume an intermediate-level understanding of Python.
The Python code should be commented, assuming a senior engineer will be reading the comments.
This is a learning exercise.
The solution doesn’t need production hardening.  Baseline error detection and handling are required.
Don’t assume any access to the email system or email client itself.
Every email provided as input should be solved. Remember, we can archive anything we can’t categorize.
Do not change the JSON format of the input file. You can make suggestions however.
Do not invent new email categories or email labels.
The input file will have :category” and “priority” set to “” (empty string)
Do not change any other parts of the email other than the “category” label and the “priority” label.
We will use OpenAI API’s as needed.  OpenAI.api_key = os.environ["OPENAI_API_KEY"] where OPENAI_API_KEY already exists.
Use an appropriate OpenAI model, and prompt as needed.
Create and use “Constants.py” for defining global constants and data structures.  Will import as needed.
