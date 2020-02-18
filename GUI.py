# Final Project
# MATHGR 5320
# Financial Risk Management & Regulation
# Professor Harvey Stein
# A Group of： Kebing Li (kl3185), Qiqi Wu (qw2273), Daniel Lee (dhl2136)
# Winter 2020

import tkinter as tk
from tkinter import messagebox
import Methods
import numpy as np

# create a class to build and manage the display
class DisplayApp:
	
	def __init__(self, width, height):

		# create a tk object, which is the root window
		self.root = tk.Tk()

		# width and height of the window
		self.initDx = width
		self.initDy = height

		# set up the geometry for the window
		self.root.geometry("%dx%d+50+30" % (self.initDx, self.initDy))

		self.root.option_add("*Font", ("Helvetica", 9))

		# set the title of the window
		self.root.title("RISK MEASURE CALCULATOR")

		# set the maximum size of the window for resizing
		self.root.maxsize(1600, 900)

		# setup the menus
		self.buildMenus()

		# build the controls
		self.buildControls()

		# build the Canvas
		self.buildCanvas()

		# bring the window to the front
		self.root.lift()

		# - do idle events here to get actual canvas size
		self.root.update_idletasks()

		# now we can ask the size of the canvas
		print(self.canvas.winfo_geometry())

		# set up the key bindings
		self.setBindings()

		# set up the application state
		self.objects = []  # list of data objects that will be drawn in the canvas
		self.size = []  # create a size list for the objects
		
		# Terminate window when close button is pressed
		self.root.protocol("WM_DELETE_WINDOW", self.handleQuit)

	def buildMenus(self):

		# create a new menu
		menu = tk.Menu(self.root)

		# set the root menu to our new menu
		self.root.config(menu=menu)

		# create a variable to hold the individual menus
		menulist = []

		# create a file menu
		filemenu = tk.Menu(menu)
		menu.add_cascade(label="File", menu=filemenu)
		menulist.append(filemenu)

		# menu text for the elements
		menutext = [['Clear Ctl-N', 'Quit Ctl-Q', 'Refresh Entries']]

		# menu callback functions
		menucmd = [[self.clearData, self.handleQuit, self.refresh]]

		# build the menu elements and callbacks
		for i in range(len(menulist)):
			for j in range(len(menutext[i])):
				if menutext[i][j] != '-':
					menulist[i].add_command(label=menutext[i][j], command=menucmd[i][j])
				else:
					menulist[i].add_separator()

	# create the canvas object
	def buildCanvas(self):

		self.canvas = tk.Canvas(self.root, width=self.initDx, height=self.initDy)
		self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

		return

	# build a frame and put controls in it
	def buildControls(self):
		
		### Control ###
		# make a control frame on the right
		rightcntlframe = tk.Frame(self.root)
		rightcntlframe.pack(side=tk.RIGHT, padx=20, pady=10, fill=tk.Y)

		# make a separator frame
		sep = tk.Frame(self.root, height=self.initDy, width=2, bd=2, relief=tk.SUNKEN)
		sep.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

		label_id = tk.Label(rightcntlframe, text="Invest Date", width=20)
		label_id.pack(side=tk.TOP, pady=1)

		self.invest_day_entry = tk.Entry(rightcntlframe)
		self.invest_day_entry.insert(tk.END, "2000-01-03")
		self.invest_day_entry.pack()

		label_ii = tk.Label(rightcntlframe, text = "inital investment ($)")
		label_ii.pack()

		self.initial_investment_entry = tk.Entry(rightcntlframe)
		self.initial_investment_entry.insert(tk.END, 1000000)
		self.initial_investment_entry.pack()

		label_th = tk.Label(rightcntlframe, text = "time horizon (days)")
		label_th.pack()

		self.time_horizon_entry = tk.Entry(rightcntlframe)
		self.time_horizon_entry.insert(tk.END, 5)
		self.time_horizon_entry.pack()

		label_VaRp = tk.Label(rightcntlframe, text = "VaR confidence level")
		label_VaRp.pack()

		self.VaRp = tk.Entry(rightcntlframe)
		self.VaRp.insert(tk.END, 0.99)
		self.VaRp.pack()

		label_ESp = tk.Label(rightcntlframe, text="ES confidence level")
		label_ESp.pack()

		self.ESp = tk.Entry(rightcntlframe)
		self.ESp.insert(tk.END, 0.975)
		self.ESp.pack()

		label_window_size = tk.Label(rightcntlframe, text="Window Size")
		label_window_size.pack()

		self.listbox1 = tk.Listbox(rightcntlframe, exportselection=0, height=3)
		self.listbox1.insert(tk.END, '2YR', '5YR', '10YR')
		self.listbox1.select_set(1)
		self.listbox1.pack(side=tk.TOP)

		label_tickers = tk.Label(rightcntlframe, text="Tickers (seperate by comma)")
		label_tickers.pack()
		self.tickers = tk.Entry(rightcntlframe)
		self.tickers.pack()

		dash2 = tk.Label(rightcntlframe, text="", width=20, font = ("Helvetica", 2))
		dash2.pack(side=tk.TOP, pady=1)

		label_weighting = tk.Label(rightcntlframe, text="Please choose one of the following Methods")
		label_weighting.pack()

		self.listbox4 = tk.Listbox(rightcntlframe, exportselection=0, height=3)
		self.listbox4.insert(tk.END, 'Parametric', 'Historical', 'Monte Carlo')
		self.listbox4.select_set(0)
		self.listbox4.pack(side=tk.TOP)

		label_weighting = tk.Label(rightcntlframe, text="Weighting Method")
		label_weighting.pack()
		self.listbox2 = tk.Listbox(rightcntlframe, exportselection=0, height=2)
		self.listbox2.insert(tk.END, 'Equal Weighting', 'Exponential Weighting')
		self.listbox2.select_set(0)
		self.listbox2.pack(side=tk.TOP)

		dash3 = tk.Label(rightcntlframe, text="", width=20, font = ("Helvetica", 2))
		dash3.pack(side=tk.TOP, pady=1)

		plot_parameter_button1 = tk.Button(rightcntlframe, text="PLOT Parameter Mu", command=self.handlePlotMu)
		plot_parameter_button1.pack()

		plot_parameter_button2 = tk.Button(rightcntlframe, text="PLOT Parameter Sigma", command=self.handlePlotSigma)
		plot_parameter_button2.pack()

		dash4 = tk.Label(rightcntlframe, text="", width=20, font = ("Helvetica", 2))
		dash4.pack(side=tk.TOP, pady=1)

		label_mc_sim_num = tk.Label(rightcntlframe, text="# of Monte Carlo Simulation")
		label_mc_sim_num.pack()
		self.mc_sim_num = tk.Entry(rightcntlframe)
		self.mc_sim_num.pack()

		label_perc_liquid = tk.Label(rightcntlframe, text="% of liquidation")
		label_perc_liquid.pack()
		self.perc_liquid = tk.Entry(rightcntlframe)
		self.perc_liquid.pack()

		self.listbox3 = tk.Listbox(rightcntlframe, exportselection=0, height=2)
		self.listbox3.insert(tk.END, 'Stock Only', 'Stock + Option')
		self.listbox3.select_set(0)
		self.listbox3.pack(side=tk.TOP)

		button = tk.Button(rightcntlframe, text="PLOT VaR", command=self.handleButton1)
		button.pack()

		button1 = tk.Button(rightcntlframe, text="PLOT ES", command=self.handleButton2)
		button1.pack()

		button2 = tk.Button(rightcntlframe, text = "Backtest", command = self.backtest)
		button2.pack()

		return

	# set bindings to two basic keyboard functionalities
	def setBindings(self):
		# bind command sequences to the root window
		self.root.bind('<Control-q>', self.handleQuit)
		self.root.bind('<Control-n>', self.clearData)

	# remove all the entries
	def refresh(self):
		self.invest_day_entry.delete(0, tk.END)
		self.initial_investment_entry.delete(0, tk.END)
		self.time_horizon_entry.delete(0, tk.END)
		self.VaRp.delete(0, tk.END)
		self.ESp.delete(0, tk.END)
		self.tickers.delete(0, tk.END)
		self.perc_liquid.delete(0,tk.END)
		self.mc_sim_num.delete(0,tk.END)

	# quit the software
	def handleQuit(self):
		print('Terminating')
		self.root.quit()
		self.root.destroy()

	# Plot Mu in parametric method
	def handlePlotMu(self):

		# get all the input/default values
		startDate = str(self.invest_day_entry.get())
		initVal = int(self.initial_investment_entry.get())
		horizon = int(self.time_horizon_entry.get())
		VaRPct = float(self.VaRp.get())
		ESPct = float(self.ESp.get())

		if len(self.tickers.get()) == 0:
			messagebox.showwarning("Warning", "Please insert your desired tickers!")

		else:
			tickerList = self.tickers.get().replace(' ','').split(',')
			winsize = self.listbox1.get(self.listbox1.curselection())
			method = self.listbox4.get(self.listbox4.curselection())
			weight = self.listbox2.get(self.listbox2.curselection())

			windowsize = 0
			if (winsize == '2YR'):
				windowsize = 2
			elif (winsize == '5YR'):
				windowsize = 5
			elif (winsize == '10YR'):
				windowsize = 10

			# if the user chooses Parametric Method to estimate risk measures
			if (method == "Parametric"):
				print('Plotting Parametric VaR')

				# if the user chooses equal weighting method
				if (weight == 'Equal Weighting'):
					print('With Equal Weighting')
					figname = Methods.plot_parameters(tickerList, startDate, windowsize, initVal,
											   equal_wgt=True)
					self.img = tk.PhotoImage(file=str(figname))
					self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

				# if the user chooses exponential weighting method
				if (weight == 'Exponential Weighting'):
					print('With Exponential Weighting')
					figname = Methods.plot_parameters(tickerList, startDate, windowsize,
											   initVal, equal_wgt=False)
					self.img = tk.PhotoImage(file=str(figname))
					self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

			else:
				messagebox.showwarning("Warning", "Plot Mu can only be used in Parametric Method!")

	# Plot Mu in parametric method
	def handlePlotSigma(self):

		# get all the input/default values
		startDate = str(self.invest_day_entry.get())
		initVal = int(self.initial_investment_entry.get())
		horizon = int(self.time_horizon_entry.get())
		VaRPct = float(self.VaRp.get())
		ESPct = float(self.ESp.get())

		if len(self.tickers.get()) == 0:
			messagebox.showwarning("Warning", "Please insert your desired tickers!")

		else:
			tickerList = self.tickers.get().replace(' ','').split(',')
			winsize = self.listbox1.get(self.listbox1.curselection())
			method = self.listbox4.get(self.listbox4.curselection())
			weight = self.listbox2.get(self.listbox2.curselection())

			windowsize = 0
			if (winsize == '2YR'):
				windowsize = 2
			elif (winsize == '5YR'):
				windowsize = 5
			elif (winsize == '10YR'):
				windowsize = 10

			# if the user chooses Parametric Method to estimate risk measures
			if (method == "Parametric"):
				print('Plotting Parametric VaR')

				# if the user chooses equal weighting method
				if (weight == 'Equal Weighting'):
					print('With Equal Weighting')
					figname = Methods.plot_parameters(tickerList, startDate, windowsize, initVal, equal_wgt=True, parameter="sigma")
					self.img = tk.PhotoImage(file=str(figname))
					self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

				# if the user chooses exponential weighting method
				if (weight == 'Exponential Weighting'):
					print('With Exponential Weighting')
					figname = Methods.plot_parameters(tickerList, startDate, windowsize, initVal, equal_wgt=False, parameter = "sigma")
					self.img = tk.PhotoImage(file=str(figname))
					self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

			else:
				messagebox.showwarning("Warning", "Plot Sigma can only be used in Parametric Method!")

	# plotting the desired VaR
	def handleButton1(self):

		# get all the input/default values
		startDate = str(self.invest_day_entry.get())
		initVal = int(self.initial_investment_entry.get())
		horizon = int(self.time_horizon_entry.get())
		VaRPct = float(self.VaRp.get())
		ESPct = float(self.ESp.get())

		if len(self.tickers.get()) == 0:
			messagebox.showwarning("Warning", "Please insert your desired tickers!")

		else:
			tickerList = self.tickers.get().replace(' ','').split(',')
			winsize = self.listbox1.get(self.listbox1.curselection())
			method = self.listbox4.get(self.listbox4.curselection())
			weight = self.listbox2.get(self.listbox2.curselection())

			windowsize = 0
			if (winsize == '2YR'):
				windowsize = 2
			elif (winsize == '5YR'):
				windowsize = 5
			elif (winsize == '10YR'):
				windowsize = 10

			# if the user chooses Parametric Method to estimate risk measures
			if (method == "Parametric"):
				print('Plotting Parametric VaR')

				# if the user chooses equal weighting method
				if (weight == 'Equal Weighting'):
					print('With Equal Weighting')
					figname = Methods.plot_gbm(tickerList, startDate, windowsize, VaRPct, ESPct, horizon, v0 = initVal, equal_wgt=True,var=True )
					self.img = tk.PhotoImage(file = str(figname))
					self.canvas.create_image(0, 0, anchor = tk.NW, image = self.img)

				# if the user chooses exponential weighting method
				if (weight == 'Exponential Weighting'):
					print('With Exponential Weighting')
					figname = Methods.plot_gbm(tickerList, startDate, windowsize, VaRPct, ESPct, horizon, v0 = initVal, equal_wgt=False, var=True)
					self.img = tk.PhotoImage(file = str(figname))
					self.canvas.create_image(0, 0, anchor = tk.NW, image = self.img)

			# if the user chooses Historical Method to estimate risk measures
			elif (method == "Historical"):
				print('Plotting Historical VaR')
				figname = Methods.plot_historical(tickerList, startDate, VaRPct, ESPct, horizon, windowsize, v0 = initVal, var=True)
				self.img = tk.PhotoImage(file=str(figname))
				self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

			# if the user chooses Monte Carlo Method to estimate risk measures
			# in this case, he/she needs to specify # of Monte Carlo Simulations
			# whether considering only stocks or a stock + option portfolio
			# in the second case, he/she needs to specify the % of liquidation
			elif (method == "Monte Carlo"):

				if len(self.mc_sim_num.get()) == 0:
					messagebox.showwarning("Warning", "Please enter # of Monte Carlo Simulation")
				else:
					npath = int(self.mc_sim_num.get())
					if npath > 5000:
						messagebox.showwarning("Warning", "The # of simulation may be too large. Will run a bit longer than expect.")
						pass
					liquidPctEntry = self.perc_liquid.get()

					optionChoice = self.listbox3.get(self.listbox3.curselection())
					print('Plotting Monte Carlo VaR')

					if len(self.perc_liquid.get()) == 0 and optionChoice == 'Stock + Option':
						messagebox.showwarning("Warning", "Please enter % of liquidation!")
					else:
						if liquidPctEntry != '' and optionChoice == 'Stock + Option':
							liquidPct = float(liquidPctEntry)/100
							figname, print_list = Methods.plot_mc(tickerList, startDate, horizon, windowsize, VaRPct, ESPct, npath, pct=liquidPct,  gbmport = True, v0 = initVal, var=True)
							self.img = tk.PhotoImage(file = str(figname))
							self.canvas.create_image(0, 0, anchor = tk.NW, image = self.img)
							# print out the amount of reduction in VaR
							print_list = [" "] + print_list
							string = '\n'.join(print_list[1:])
							messagebox.showinfo("VaR Reduction Amount", string)

						elif optionChoice == 'Stock Only':
							figname, print_list = Methods.plot_mc(tickerList, '2000-01-03', horizon, windowsize, VaRPct, ESPct, npath, pct=None,  gbmport = True, v0 = initVal, var=True)
							self.img = tk.PhotoImage(file = str(figname))
							self.canvas.create_image(0, 0, anchor = tk.NW, image = self.img)

	# plot the desire ES
	def handleButton2(self):
		# get all the input/default values
		startDate = str(self.invest_day_entry.get())
		initVal = int(self.initial_investment_entry.get())
		horizon = int(self.time_horizon_entry.get())
		VaRPct = float(self.VaRp.get())
		ESPct = float(self.ESp.get())

		if len(self.tickers.get()) == 0:
			messagebox.showwarning("Warning", "Please insert your desired tickers!")

		else:
			tickerList = self.tickers.get().replace(' ','').split(',')
			winsize = self.listbox1.get(self.listbox1.curselection())
			method = self.listbox4.get(self.listbox4.curselection())
			weight = self.listbox2.get(self.listbox2.curselection())

			windowsize = 0
			if (winsize == '2YR'):
				windowsize = 2
			elif (winsize == '5YR'):
				windowsize = 5
			elif (winsize == '10YR'):
				windowsize = 10

			# if the user chooses Parametric Method to estimate risk measures
			if (method == "Parametric"):
				print('Plotting Parametric VaR')

				# if the user chooses equal weighting method
				if (weight == 'Equal Weighting'):
					print('With Equal Weighting')
					figname = Methods.plot_gbm(tickerList, startDate, windowsize, VaRPct, ESPct, horizon, v0=initVal,
											   equal_wgt=True, var=False)
					self.img = tk.PhotoImage(file=str(figname))
					self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

				# if the user chooses exponential weighting method
				if (weight == 'Exponential Weighting'):
					print('With Exponential Weighting')
					figname = Methods.plot_gbm(tickerList, startDate, windowsize, VaRPct, ESPct, horizon, v0=initVal,
											   equal_wgt=False, var=False)
					self.img = tk.PhotoImage(file=str(figname))
					self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

			# if the user chooses Historical Method to estimate risk measures
			elif (method == "Historical"):
				print('Plotting Historical VaR')
				figname = Methods.plot_historical(tickerList, startDate, VaRPct, ESPct, horizon, windowsize,
												  v0=initVal, var=False)
				self.img = tk.PhotoImage(file=str(figname))
				self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

			# if the user chooses Monte Carlo Method to estimate risk measures
			# in this case, he/she needs to specify # of Monte Carlo Simulations
			# whether considering only stocks or a stock + option portfolio
			# in the second case, he/she needs to specify the % of liquidation
			elif (method == "Monte Carlo"):

				if len(self.mc_sim_num.get()) == 0:
					messagebox.showwarning("Warning", "Please enter # of Monte Carlo Simulation")
				else:
					npath = int(self.mc_sim_num.get())
					if npath > 5000:
						messagebox.showwarning("Warning",
											   "The # of simulation may be too large. Will run a bit longer than expect.")
					pass
					liquidPctEntry = self.perc_liquid.get()

					optionChoice = self.listbox3.get(self.listbox3.curselection())
					print('Plotting Monte Carlo VaR')

					if len(self.perc_liquid.get()) == 0 and optionChoice == 'Stock + Option':
						messagebox.showwarning("Warning", "Please enter % of liquidation!")
					else:
						if liquidPctEntry != '' and optionChoice == 'Stock + Option':
							messagebox.showwarning("Warning", "Cannot compute ES for stock option portfolio")

						elif optionChoice == 'Stock Only':
							figname, print_list = Methods.plot_mc(tickerList, startDate, horizon, windowsize, VaRPct, ESPct,
													  npath, pct=None, gbmport=True, v0=initVal, var=False)
							self.img = tk.PhotoImage(file=str(figname))
							self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

	# plot the backtest plot based on the user selection
	def backtest(self):

		# get all the input/default values
		startDate = str(self.invest_day_entry.get())
		initVal = int(self.initial_investment_entry.get())
		horizon = int(self.time_horizon_entry.get())
		VaRPct = float(self.VaRp.get())
		ESPct = float(self.ESp.get())

		if len(self.tickers.get()) == 0:
			messagebox.showwarning("Warning", "Please insert your desired tickers!")

		else:
			tickerList = self.tickers.get().replace(' ','').split(',')
			winsize = self.listbox1.get(self.listbox1.curselection())
			method = self.listbox4.get(self.listbox4.curselection())
			weight = self.listbox2.get(self.listbox2.curselection())

			windowsize = 0
			if (winsize == '2YR'):
				windowsize = 2
			elif (winsize == '5YR'):
				windowsize = 5
			elif (winsize == '10YR'):
				windowsize = 10

			# if the user chooses Parametric Method to estimate risk measures
			if (method == "Parametric"):
				print('Plotting Parametric VaR')

				# if the user chooses equal weighting method
				if (weight == 'Equal Weighting'):
					print('With Equal Weighting')
					figname = Methods.back_test(tickerList, startDate, VaRPct, ESPct, horizon, windowsize, 0, method="P", v0 = initVal, equal_wgt=True)
					self.img = tk.PhotoImage(file = str(figname))
					self.canvas.create_image(0, 0, anchor = tk.NW, image = self.img)

				# if the user chooses exponential weighting method
				if (weight == 'Exponential Weighting'):
					print('With Exponential Weighting')
					figname = Methods.back_test(tickerList, startDate, VaRPct, ESPct, horizon, windowsize, 0, method="P", v0 = initVal, equal_wgt=False)
					self.img = tk.PhotoImage(file = str(figname))
					self.canvas.create_image(0, 0, anchor = tk.NW, image = self.img)

			# if the user chooses Historical Method to estimate risk measures
			elif (method == "Historical"):
				print('Plotting Historical VaR')
				figname = Methods.back_test(tickerList, startDate, VaRPct, ESPct, horizon, windowsize, 0, method="H",
											v0=initVal, equal_wgt=True)
				self.img = tk.PhotoImage(file=str(figname))
				self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

			# if the user chooses Monte Carlo Method to estimate risk measures
			# in this case, he/she needs to specify # of Monte Carlo Simulations
			# whether considering only stocks or a stock + option portfolio
			# in the second case, he/she needs to specify the % of liquidation
			elif (method == "Monte Carlo"):

				if len(self.mc_sim_num.get()) == 0:
					messagebox.showwarning("Warning", "Please enter # of Monte Carlo Simulation")
				else:
					npath = int(self.mc_sim_num.get())
					if npath > 5000:
						messagebox.showwarning("Warning", "The # of simulation may be too large. Will run a bit longer than expect.")
						pass
					liquidPctEntry = self.perc_liquid.get()

					optionChoice = self.listbox3.get(self.listbox3.curselection())
					print('Plotting Monte Carlo VaR')

					if len(self.perc_liquid.get()) == 0 and optionChoice == 'Stock + Option':
						messagebox.showwarning("Warning", "Please enter % of liquidation!")
					else:
						if liquidPctEntry != '' and optionChoice == 'Stock + Option':
							messagebox.showwarning("Warning", "Cannot backtest on stock option portfolio")

						elif optionChoice == 'Stock Only':
							figname = Methods.back_test(tickerList, startDate, VaRPct, ESPct, horizon, windowsize, npath,
														method="M", v0=initVal, equal_wgt=True)
							self.img = tk.PhotoImage(file = str(figname))
							self.canvas.create_image(0, 0, anchor = tk.NW, image = self.img)

	def handleMenuCmd1(self):
		print('handling menu command 1')

	# clear the canvas when pressing Control-n
	def clearData(self, event=None):
		self.canvas.delete("all")
		print("Start a new round")

	def main(self):
		print('Entering main loop')
		self.root.mainloop()


if __name__ == "__main__":
	dapp = DisplayApp(1000, 700)
	dapp.main()