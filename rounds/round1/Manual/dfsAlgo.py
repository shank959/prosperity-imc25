exchangeRates = {
    "Snow": {"Snow": 1, "Piz": 1.45, "Si": 0.52, "Sea": 0.72},
    "Piz":  {"Snow": 0.7, "Piz": 1, "Si": 0.31, "Sea": 0.48},
    "Si":  {"Snow": 1.95, "Piz": 3.1, "Si": 1, "Sea": 1.49},
    "Sea": {"Snow": 1.34, "Piz": 1.98, "Si": 0.64, "Sea": 1},
}

max_depth = 5
initial_currency = "Sea"
initial_amount = 500
profitable_paths = []

def dfs(path, current_currency, amount, depth):
    if depth > max_depth:
        return
    if depth >= 1 and current_currency == "Sea" and amount > initial_amount:
        profitable_paths.append((path[:], amount))
    
    for next_currency, rate in exchangeRates[current_currency].items():
        next_amount = amount * rate
        path.append(next_currency)
        dfs(path, next_currency, next_amount, depth + 1)
        path.pop()

dfs(["Sea"], "Sea", initial_amount, 0)

profitable_paths.sort(key=lambda x: x[1], reverse=True)

for path, final_amount in profitable_paths[:10]:
    print(f"Path: {' -> '.join(path)}, Final: {final_amount:.2f}")
    

