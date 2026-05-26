import random
import datetime

def generate_agent_training_data(count=5000):
    dataset = []
    
    # Tool 1: Math Operators
    ops = {
        "plus": "+", "add": "+", "sum": "+",
        "minus": "-", "subtract": "-", "difference": "-",
        "times": "*", "multiply": "*", "product": "*",
        "divided by": "/", "divide": "/"
    }

    for i in range(count):
        # 50% Math, 50% Time
        if i % 2 == 0:
            # --- MATH CASE ---
            a, b = random.randint(1, 10), random.randint(1, 10)
            word_op, symbol = random.choice(list(ops.items()))
            
            # Ensure no division by zero
            if symbol == "/" and b == 0: b = 1 
            
            # Format: User -> Thought -> Action -> Observation -> Answer
            # Note: We include the Observation in training so the model 
            # learns how to react to the result.
            res = round(eval(f"{a}{symbol}{b}"), 2)
            
            entry = (
                f"User: {a} {word_op} {b} = ? "
                f"Thought: need arithmetic. "
                f"Action: CALC( {a} {symbol} {b} ) "
                f"Answer: {res} <|end|>"
            )
        else:
            # --- TIME CASE ---
            queries = ["time please", "current time", "hour please"]
            q = random.choice(queries)
            
            # Fake a time for training purposes
            h, m = random.randint(0, 23), random.randint(0, 59)
            fake_time = f"{h:02d}:{m:02d}"
            
            entry = (
                f"User: {q} "
                f"Thought: need clock. "
                f"Action: GET_TIME() "
                f"Answer: {fake_time} <|end|>"
            )
        
        dataset.append(entry)
    
    return dataset

# Generate and save
train_data = generate_agent_training_data(10000)
with open("../../data/agent_data.txt", "w") as f:
    f.write("\n\n".join(train_data))

print(f"Generated {len(train_data)} examples in ../../data/agent_data.txt")