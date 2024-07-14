import sqlite3

def review_feedback():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM feedback")
    rows = cursor.fetchall()
    conn.close()
    
    for row in rows:
        print(f"ID: {row[0]}")
        print(f"URL: {row[1]}")
        print(f"Prediction: {row[2]}")
        print(f"Feedback: {row[3]}")
        print(f"Comments: {row[4]}")
        print("")

if __name__ == "__main__":
    review_feedback()
