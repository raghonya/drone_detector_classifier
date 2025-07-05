import	json, signal, websockets, asyncio, inspect
from	datetime import datetime
from	colorama import Style, init, Fore

init()

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
caller_file = ""

def sig_handler(sig: int, frame=None) -> None:
	global server

	LOG_DBG(f"{sig.name} received, Closing the server", level=INFO)
	if (server is not None):
		server.close()

def LOG_DBG(message: str, level: str = INFO):
	"""
	Print's message 
	- if ENABLE_LOG_FILES is True, prints to each server's own log file
	- Otherwise` prints to the console
	
	Args:
		message (str): Message to print.
		level (str): Message's debug level (INIT/INFO/DEBUG/ERROR).
		location (_io.TextIOWrapper): file to write

	Returns: Nothing
	"""
	global	caller_file
	global	fd_logfile

	if level in LOG:
		if (False):
			log_path = f"/tmp/{caller_file}.log"
			with open(log_path, 'a') as fd_logfile:
				fd_logfile.write(f"[{datetime.now()}] [ {level} ] {message}\n")
			
			# # # for limit of 200 lines
			#with open(log_path, 'r+') as fd_logfile:
				#lines = fd_logfile.readlines()
				#if len(lines) > 200:
				#	lines = lines[1:]
				#lines.append(f"[{datetime.now()}] [ {level} ] {message}\n")
				#fd_logfile.seek(0)
				#fd_logfile.truncate()
				#fd_logfile.writelines(lines)

				print("Written in file")
		else: # ENABLE_LOG_FILES is disabled, printing to contole
			print(f"{color[level]}[{datetime.now()}] ", end="")
			print(f"[ {level} ] {message}", end="")
			print(f"{Style.RESET_ALL}")


async def sendError(websocket, message):
	"""
	Print's the error and sends it to the client's side.

	Args:
		websocket : Websocket of the client.
		message (str): Error message.

	Returns: Nothing
	"""
	LOG_DBG(message, ERROR)
	await websocket.send(json.dumps({"Status": "Error", "Payload": f"{message}"}))

async def startServer(port: int, handler: callable) -> None:
	"""
	Starts server on given port.

	Args:
		port (int): Port of the server
		handler (callable): Request handler function for clients
	Returns: Nothing
	"""
	global server

	server = await websockets.serve(handler, "0.0.0.0", port, ping_timeout=60)
	LOG_DBG(f"Server started on ws://0.0.0.0:{port}", INIT)
	loop = asyncio.get_event_loop()
	loop.add_signal_handler(signal.SIGINT, sig_handler, (signal.SIGINT))
	loop.add_signal_handler(signal.SIGTERM, sig_handler, (signal.SIGTERM))
	await server.wait_closed()
	LOG_DBG("End of main() function", INFO)
