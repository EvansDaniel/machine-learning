import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from time import sleep
from time import time
import json
import traceback
import datetime
import os
import sys

UNUSED_BILL_TEXT = ''

using_python3 = sys.version_info[0] >= 3
if not using_python3:
	print('You must use python 3. Run with python3 <filename>')
	sys.exit(1)

delay = .1  # time to wait for scrolling (comments)

def open_file(filename, *args, **kwargs):
	# Create file if not exists
    open(filename, 'a').close()
    # Encapsulate the low-level file descriptor in a python file object
    return open(filename, *args, **kwargs)

def get_bill_text_from_file(url, filename):
	try:
		bill_file = open_file(filename, 'r+')
		try:
			bills = json.load(bill_file)
			for bill_dict in bills:
				if url == bill_dict['url']:
					return bill_dict['text']
			return None
		except:
			if os.stat(filename).st_size != 0:
				print(traceback.format_exc())
			return None
	finally:
		if bill_file:
			bill_file.close()

def parse_json_line(parser_line):
	return [x.strip() for x in parser_line.split('.') if x.strip()]

def get_item_from_json(json_obj, parser_line):
	parsed_line = parse_json_line(parser_line)
	if len(parsed_line) == 0:
		return json_obj
	for i in range(len(parsed_line)):
		if parsed_line[i] in json_obj:
			item = json_obj[parsed_line[i]]
		else:
			item = None
			print('No such key', '.'.join(parsed_line[:i+1]))
	return item

def add_item_to_json(item, open_file, parser_line):
	try:
		# Array of bill_dicts (like defined above)
		json_obj = json.load(open_file)
		json_item = get_item_from_json(json_obj, parser_line)
		if isinstance(json_item, list):
			json_item.append(item)
		else:
			print('item retrieved from json is not a list')
			return None
	except:
		print('Failed to load', filename)
		print(traceback.format_exc())
		json_item = [item]
	return json_item

def save_bill_details(url, bill_text, filename):
	print('saving to', filename)
	bill_dict = { 'url': url, 'text': bill_text }
	try:
		bill_file = open_file(filename, 'r+')
		json_bill = add_item_to_json(bill_dict, bill_file, '.')
		# Seek to beginning and truncate the file
		bill_file.seek(0)
		bill_file.truncate()
		json.dump(json_bill, bill_file)
	finally:
		if bill_file:
			bill_file.close()

def get_and_save_bill_text(url, filename):
	bill_text = get_bill_text_from_file(url, filename)
	if bill_text is not None:
		print('Loaded from cache')
		return bill_text
	print('Did not find cached bill text. Loading from url')
	bill_container_class = '.generated-html-container'
	driver = webdriver.Chrome()
	driver.get(url)
	#if driver.find_element_by_css_selector('.legDetail').text == UNUSED_BILL_TEXT
	bill_text = driver.find_element_by_css_selector(bill_container_class).text
	driver.close()
	save_bill_details(url, bill_text, filename)
	return bill_text

def get_last_digit(num):
	while(num >= 10):
		num = num % 10
	return num

def get_number_suffix(num):
	num = int(num)
	last_num = get_last_digit(num)
	if last_num == 1:
		return 'st'
	if last_num == 3:
		return 'rd'
	return 'th'

def generate_url(congress_num, bill_id):
	num_suffix = get_number_suffix(congress_num)
	return 'https://www.congress.gov/bill/'+congress_num+num_suffix+'-congress/house-bill/'+bill_id+ '/text'

if __name__ == "__main__":
	congress_num = input('Enter congress number ')
	bill_id = input('Enter bill number ')
	filename = 'congress-bills.json'
	url = generate_url(congress_num, bill_id)
	bill_text = get_and_save_bill_text(url, filename)
	#print(bill_text)