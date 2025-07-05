import json, asyncio, websockets
from copy import deepcopy
from colorama import Fore
import aiohttp
import numpy as np
#from predict_drone_type import predict_type
# local libraries
import Hardware.Server.lib as lib
from Hardware.Server.lib import LOG_DBG

########################################
# Debugging

INIT = "INIT"
INFO = "INFO"
DEBUG = "DEBUG"
ERROR = "ERROR"

color = {
	INIT: Fore.GREEN,
	INFO: Fore.YELLOW,
	DEBUG: Fore.WHITE, 
	ERROR: Fore.RED
}

LOG = [INIT, INFO, ERROR]

server = None
waiting_event = None
def sig_handler(sig: int, frame=None) -> None:
	global server

	LOG_DBG(f"{sig.name} received, Closing the server", level=INFO)
	if (server):
		if (waiting_event):
			waiting_event.set()
		server.close()

########################################

#async def ask_ai(prompt):
#	url = 'https://openrouter.ai/api/v1/chat/completions'
#	headers = {
#		'Authorization': 'Bearer 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',
#		'Content-Type': 'application/json',
#		'HTTP-Referer': 'http://yourdomain.com',
#		'X-Title': 'My OpenRouter Test'
#	}

#	body = {
#		"model": "mistralai/mistral-7b-instruct",
#		"messages": [
#			{"role": "system", "content": "You are a helpful AI assistant."},
#			{"role": "user", "content": prompt}
#		],
#		"temperature": 0.8,
#		"max_tokens": 300
#	}

#	async with aiohttp.ClientSession() as session:
#		async with session.post(url, headers=headers, json=body) as resp:
#			data = await resp.json()
#			print(json.dumps(data, indent=2))
#			return data['choices'][0]['message']['content']

## Example usage
#prompt = "Tell me about stars"

# # Commands and values

freq_range_min = 5690
freq_range_max = 5910
data_lock = None
ready_lock = None
ready_to_send = False
sendStatusOK = {"Status": 'OK', "Payload": ''}
payload = {"Frequency": [0], "Amplitude": [0]}
sendValue = deepcopy(sendStatusOK)
connectedClient = None
freqs = []
amplitudes = []
sender_ws = None
receiver_ws = None
sender_is_connected = False
drone_type = ""

def undbfs(db_vals: list):
    db_vals = np.asarray(db_vals, dtype=np.float64)
    db_vals = np.where(np.isfinite(db_vals), db_vals, -120.0)

    power = 10.0 ** (db_vals / 10.0)
    return np.maximum(power, 1e-12)

def detected_drone_type(freq: list, amplitudes: list) -> str:
	#spectrum_power = undbfs(amplitudes)
	#drone, conf = predict_type(freq, spectrum_power)  # <= именно spectrum_power!
	#if conf > 0.75 and drone != "None":

	#	print(f"{drone} {conf:.3f}")
	return ""

def detect_drone(amplitudes: list) -> bool:
    count = 0
    for amp in amplitudes:
        if amp >= -70:
            count += 1
            if count >= 3:
                return True  # Drone detected
        else:
            count = 0  # Reset the count if the condition is broken
    return False  # No drone detected

async def command_GetValue(websocket = None, data = {}):
	global freqs
	global amplitudes
	global drone_type

	#LOG_DBG("GET COMMAND", INIT)
	#LOG_DBG(drone_type, INIT)
	await waiting_event.wait()
	await websocket.send(
		json.dumps({
			"Status": "OK", 
			"Payload": json.dumps({
				"Frequency": freqs,
				"Amplitude": amplitudes,
				"Detected": detect_drone(amplitudes),
				"Type": drone_type,
				"Location": json.dumps({
					"lat": 40.79725734499209,
					"lng": 44.52214384246903,
					"radius": 1000
				})
			})
		})
	)
	freqs.clear()
	amplitudes.clear()
	waiting_event.clear()
	drone_type = ""
	pass

async def command_GetData(websocket = None, data = {}):
	global drone_type
	global freqs
	global amplitudes
	#LOG_DBG(data, INFO)
	tmp = json.loads(data["Payload"])
	if (tmp["Type"] and tmp["Type"] != "None"):
		drone_type = tmp["Type"]
	LOG_DBG(f'type: {drone_type}', INFO)

	freqs.extend(tmp["Frequency"])
	amplitudes.extend(tmp["Amplitude"])
	#LOG_DBG(f"freq: {len(freqs)}, ampl: {len(amplitudes)}")
	if (freqs and abs(freqs[-1] - freq_range_max) <= 10):
		waiting_event.set()

	#freqs = payload["Frequency"]
	#amplitudes = payload["Amplitude"]

	#LOG_DBG(len(payload["Frequency"]))
	#LOG_DBG(f'freqs: {payload["Frequency"][:10]}')
	#LOG_DBG(f'amplitudes: {payload["Amplitude"][:10]}')	
	pass

async def command_Limits(websocket = None, data = {}):
	global freq_range_min, freq_range_max
	tmp = json.loads(data["Payload"])
	freq_range_min = tmp["Min"]
	freq_range_max = tmp["Max"]
	LOG_DBG(f"{freq_range_min} and {freq_range_max}")

cmdHandlers = {
	"Data": command_GetData,
	"Limits": command_Limits,
	"Get": command_GetValue
} 

# # WebSocket request handler
async def clientConnectionHandler(websocket, path):
	global sendValue
	global sender_is_connected
	global receiver_ws
	global sender_ws

	if (path == "/data"):
		sender_is_connected = True
		sender_ws = websocket
	if (path == "/receive"):
		if (sender_is_connected):
			receiver_ws = websocket
		else:
			await websocket.close()
	LOG_DBG(f"New client connected [{websocket.remote_address} {path}]", INIT)
	try:
		async for message in websocket:
			data = json.loads(message)
			#LOG_DBG(f"data is: '{data['CMD']}'", INFO)
			await cmdHandlers[data["CMD"]](websocket, data)
	except websockets.exceptions.ConnectionClosed:
		import traceback
		LOG_DBG(traceback.format_exc(), ERROR)
		LOG_DBG(f"Client Disconnected {path}", ERROR)
	except Exception as e:
		import traceback
		LOG_DBG(traceback.format_exc(), ERROR)
		await lib.sendError(websocket, e)
	finally:
		if (path == "/data"):
			sender_ws = None
			if receiver_ws: await receiver_ws.close()
		receiver_ws = None
		LOG_DBG(f"End of connection {path}", INFO)
# # WebSocket request handler


# # Main function call
import signal
async def main():
	global waiting_event, data_lock, ready_lock
	waiting_event = asyncio.Event()
	#data_lock = asyncio.Lock()
	#ready_lock = asyncio.Lock()
	#await ask_ai(prompt)
	global server

	server = await websockets.serve(clientConnectionHandler, "0.0.0.0", 7000, ping_timeout=60)
	LOG_DBG(f"Server started on ws://0.0.0.0:{7000}", INIT)
	loop = asyncio.get_event_loop()
	loop.add_signal_handler(signal.SIGINT, sig_handler, (signal.SIGINT))
	loop.add_signal_handler(signal.SIGTERM, sig_handler, (signal.SIGTERM))
	await server.wait_closed()
	LOG_DBG("End of main() function", INFO)

	#await lib.startServer(7000, clientConnectionHandler)

if __name__ == "__main__":
	asyncio.run(main())
