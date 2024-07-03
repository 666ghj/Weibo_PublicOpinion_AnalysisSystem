(function(jQuery) {

    "use strict";

// for apexchart
function apexChartUpdate(chart, detail) {
    let color = getComputedStyle(document.documentElement).getPropertyValue('--dark');
    if (detail.dark) {
      color = getComputedStyle(document.documentElement).getPropertyValue('--white');
    }
    chart.updateOptions({
      chart: {
        foreColor: color
      }
    })
  }
  
// for am chart
function amChartUpdate(chart, detail) {
  // let color = getComputedStyle(document.documentElement).getPropertyValue('--dark');
  if (detail.dark) {
    // color = getComputedStyle(document.documentElement).getPropertyValue('--white');
    chart.stroke = am4core.color(getComputedStyle(document.documentElement).getPropertyValue('--white'));
  }
  chart.validateData();
}

/*---------------------------------------------------------------------
   Apex Charts
-----------------------------------------------------------------------*/
if (jQuery("#apex-basic").length) {
  options = {
    chart: {
      height: 350,
      type: "line",
      zoom: {
        enabled: !1
      }
    },
    colors: ["#4788ff"],
    series: [{
      name: "Desktops",
      data: [10, 41, 35, 51, 49, 62, 69, 91, 148]
    }],
    dataLabels: {
      enabled: !1
    },
    stroke: {
      curve: "straight"
    },
    title: {
      text: "Product Trends by Month",
      align: "left"
    },
    grid: {
      row: {
        colors: ["#f3f3f3", "transparent"],
        opacity: .5
      }
    },
    xaxis: {
      categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    }
  };
  if(typeof ApexCharts !== typeof undefined){
    (chart = new ApexCharts(document.querySelector("#apex-basic"), options)).render()
    const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })
  }
}
if (jQuery("#apex-line-area").length) {
  options = {
    chart: {
      height: 350,
      type: "area"
    },
    dataLabels: {
      enabled: !1
    },
    stroke: {
      curve: "smooth"
    },
    colors: ["#4788ff", "#ff4b4b"],
    series: [{
      name: "series1",
      data: [31, 40, 28, 51, 42, 109, 100]
    }, {
      name: "series2",
      data: [11, 32, 45, 32, 34, 52, 41]
    }],
    xaxis: {
      type: "datetime",
      categories: ["2018-09-19T00:00:00", "2018-09-19T01:30:00", "2018-09-19T02:30:00", "2018-09-19T03:30:00", "2018-09-19T04:30:00", "2018-09-19T05:30:00", "2018-09-19T06:30:00"]
    },
    tooltip: {
      x: {
        format: "dd/MM/yy HH:mm"
      }
    }
  };
  (chart = new ApexCharts(document.querySelector("#apex-line-area"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#apex-bar").length) {
  options = {
    chart: {
      height: 350,
      type: "bar"
    },
    plotOptions: {
      bar: {
        horizontal: !0
      }
    },
    dataLabels: {
      enabled: !1
    },
    colors: ["#4788ff"],
    series: [{
      data: [470, 540, 580, 690, 1100, 1200, 1380]
    }],
    xaxis: {
      categories: ["Netherlands", "Italy", "France", "Japan", "United States", "China", "Germany"]
    }
  };
  (chart = new ApexCharts(document.querySelector("#apex-bar"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#apex-column").length) {
  options = {
    chart: {
      height: 350,
      type: "bar"
    },
    plotOptions: {
      bar: {
        horizontal: !1,
        columnWidth: "55%",
        endingShape: "rounded"
      }
    },
    dataLabels: {
      enabled: !1
    },
    stroke: {
      show: !0,
      width: 2,
      colors: ["transparent"]
    },
    colors: ["#4788ff", "#37e6b0", "#ff4b4b"],
    series: [{
      name: "Net Profit",
      data: [44, 55, 57, 56, 61, 58]
    }, {
      name: "Revenue",
      data: [76, 85, 101, 98, 87, 105]
    }, {
      name: "Free Cash Flow",
      data: [35, 41, 36, 26, 45, 48]
    }],
    xaxis: {
      categories: ["Feb", "Mar", "Apr", "May", "Jun", "Jul"]
    },
    yaxis: {
      title: {
        text: "$ (thousands)"
      }
    },
    fill: {
      opacity: 1
    },
    tooltip: {
      y: {
        formatter: function(e) {
          return "$ " + e + " thousands"
        }
      }
    }
  };
  (chart = new ApexCharts(document.querySelector("#apex-column"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#apex-mixed-chart").length) {
  options = {
    chart: {
      height: 350,
      type: "line",
      stacked: !1
    },
    stroke: {
      width: [0, 2, 5],
      curve: "smooth"
    },
    plotOptions: {
      bar: {
        columnWidth: "50%"
      }
    },
    colors: ["#ff4b4b", "#37e6b0", "#4788ff"],
    series: [{
      name: "Facebook",
      type: "column",
      data: [23, 11, 22, 27, 13, 22, 37, 21, 44, 22, 30]
    }, {
      name: "Vine",
      type: "area",
      data: [44, 55, 41, 67, 22, 43, 21, 41, 56, 27, 43]
    }, {
      name: "Dribbble",
      type: "line",
      data: [30, 25, 36, 30, 45, 35, 64, 52, 59, 36, 39]
    }],
    fill: {
      opacity: [.85, .25, 1],
      gradient: {
        inverseColors: !1,
        shade: "light",
        type: "vertical",
        opacityFrom: .85,
        opacityTo: .55,
        stops: [0, 100, 100, 100]
      }
    },
    labels: ["01/01/2003", "02/01/2003", "03/01/2003", "04/01/2003", "05/01/2003", "06/01/2003", "07/01/2003", "08/01/2003", "09/01/2003", "10/01/2003", "11/01/2003"],
    markers: {
      size: 0
    },
    xaxis: {
      type: "datetime"
    },
    yaxis: {
      min: 0
    },
    tooltip: {
      shared: !0,
      intersect: !1,
      y: {
        formatter: function(e) {
          return void 0 !== e ? e.toFixed(0) + " views" : e
        }
      }
    },
    legend: {
      labels: {
        useSeriesColors: !0
      },
      markers: {
        customHTML: [function() {
          return ""
        }, function() {
          return ""
        }, function() {
          return ""
        }]
      }
    }
  };
  (chart = new ApexCharts(document.querySelector("#apex-mixed-chart"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#apex-candlestick-chart").length) {
  options = {
    chart: {
      height: 350,
      type: "candlestick"
    },
    colors: ["#4788ff", "#37e6b0", "#37e6b0"],
    series: [{
      data: [{
        x: new Date(15387786e5),
        y: [6629.81, 6650.5, 6623.04, 6633.33]
      }, {
        x: new Date(15387804e5),
        y: [6632.01, 6643.59, 6620, 6630.11]
      }, {
        x: new Date(15387822e5),
        y: [6630.71, 6648.95, 6623.34, 6635.65]
      }, {
        x: new Date(1538784e6),
        y: [6635.65, 6651, 6629.67, 6638.24]
      }, {
        x: new Date(15387858e5),
        y: [6638.24, 6640, 6620, 6624.47]
      }, {
        x: new Date(15387876e5),
        y: [6624.53, 6636.03, 6621.68, 6624.31]
      }, {
        x: new Date(15387894e5),
        y: [6624.61, 6632.2, 6617, 6626.02]
      }, {
        x: new Date(15387912e5),
        y: [6627, 6627.62, 6584.22, 6603.02]
      }, {
        x: new Date(1538793e6),
        y: [6605, 6608.03, 6598.95, 6604.01]
      }, {
        x: new Date(15387948e5),
        y: [6604.5, 6614.4, 6602.26, 6608.02]
      }, {
        x: new Date(15387966e5),
        y: [6608.02, 6610.68, 6601.99, 6608.91]
      }, {
        x: new Date(15387984e5),
        y: [6608.91, 6618.99, 6608.01, 6612]
      }, {
        x: new Date(15388002e5),
        y: [6612, 6615.13, 6605.09, 6612]
      }, {
        x: new Date(1538802e6),
        y: [6612, 6624.12, 6608.43, 6622.95]
      }, {
        x: new Date(15388038e5),
        y: [6623.91, 6623.91, 6615, 6615.67]
      }, {
        x: new Date(15388056e5),
        y: [6618.69, 6618.74, 6610, 6610.4]
      }, {
        x: new Date(15388074e5),
        y: [6611, 6622.78, 6610.4, 6614.9]
      }, {
        x: new Date(15388092e5),
        y: [6614.9, 6626.2, 6613.33, 6623.45]
      }, {
        x: new Date(1538811e6),
        y: [6623.48, 6627, 6618.38, 6620.35]
      }, {
        x: new Date(15388128e5),
        y: [6619.43, 6620.35, 6610.05, 6615.53]
      }, {
        x: new Date(15388146e5),
        y: [6615.53, 6617.93, 6610, 6615.19]
      }, {
        x: new Date(15388164e5),
        y: [6615.19, 6621.6, 6608.2, 6620]
      }, {
        x: new Date(15388182e5),
        y: [6619.54, 6625.17, 6614.15, 6620]
      }, {
        x: new Date(153882e7),
        y: [6620.33, 6634.15, 6617.24, 6624.61]
      }, {
        x: new Date(15388218e5),
        y: [6625.95, 6626, 6611.66, 6617.58]
      }, {
        x: new Date(15388236e5),
        y: [6619, 6625.97, 6595.27, 6598.86]
      }, {
        x: new Date(15388254e5),
        y: [6598.86, 6598.88, 6570, 6587.16]
      }, {
        x: new Date(15388272e5),
        y: [6588.86, 6600, 6580, 6593.4]
      }, {
        x: new Date(1538829e6),
        y: [6593.99, 6598.89, 6585, 6587.81]
      }, {
        x: new Date(15388308e5),
        y: [6587.81, 6592.73, 6567.14, 6578]
      }, {
        x: new Date(15388326e5),
        y: [6578.35, 6581.72, 6567.39, 6579]
      }, {
        x: new Date(15388344e5),
        y: [6579.38, 6580.92, 6566.77, 6575.96]
      }, {
        x: new Date(15388362e5),
        y: [6575.96, 6589, 6571.77, 6588.92]
      }, {
        x: new Date(1538838e6),
        y: [6588.92, 6594, 6577.55, 6589.22]
      }, {
        x: new Date(15388398e5),
        y: [6589.3, 6598.89, 6589.1, 6596.08]
      }, {
        x: new Date(15388416e5),
        y: [6597.5, 6600, 6588.39, 6596.25]
      }, {
        x: new Date(15388434e5),
        y: [6598.03, 6600, 6588.73, 6595.97]
      }, {
        x: new Date(15388452e5),
        y: [6595.97, 6602.01, 6588.17, 6602]
      }, {
        x: new Date(1538847e6),
        y: [6602, 6607, 6596.51, 6599.95]
      }, {
        x: new Date(15388488e5),
        y: [6600.63, 6601.21, 6590.39, 6591.02]
      }, {
        x: new Date(15388506e5),
        y: [6591.02, 6603.08, 6591, 6591]
      }, {
        x: new Date(15388524e5),
        y: [6591, 6601.32, 6585, 6592]
      }, {
        x: new Date(15388542e5),
        y: [6593.13, 6596.01, 6590, 6593.34]
      }, {
        x: new Date(1538856e6),
        y: [6593.34, 6604.76, 6582.63, 6593.86]
      }, {
        x: new Date(15388578e5),
        y: [6593.86, 6604.28, 6586.57, 6600.01]
      }, {
        x: new Date(15388596e5),
        y: [6601.81, 6603.21, 6592.78, 6596.25]
      }, {
        x: new Date(15388614e5),
        y: [6596.25, 6604.2, 6590, 6602.99]
      }, {
        x: new Date(15388632e5),
        y: [6602.99, 6606, 6584.99, 6587.81]
      }, {
        x: new Date(1538865e6),
        y: [6587.81, 6595, 6583.27, 6591.96]
      }, {
        x: new Date(15388668e5),
        y: [6591.97, 6596.07, 6585, 6588.39]
      }, {
        x: new Date(15388686e5),
        y: [6587.6, 6598.21, 6587.6, 6594.27]
      }, {
        x: new Date(15388704e5),
        y: [6596.44, 6601, 6590, 6596.55]
      }, {
        x: new Date(15388722e5),
        y: [6598.91, 6605, 6596.61, 6600.02]
      }, {
        x: new Date(1538874e6),
        y: [6600.55, 6605, 6589.14, 6593.01]
      }, {
        x: new Date(15388758e5),
        y: [6593.15, 6605, 6592, 6603.06]
      }]
    }],
    title: {
      text: "CandleStick Chart",
      align: "left"
    },
    xaxis: {
      type: "datetime"
    },
    yaxis: {
      tooltip: {
        enabled: !0
      }
    }
  };
  (chart = new ApexCharts(document.querySelector("#apex-candlestick-chart"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#apex-bubble-chart").length) {
  function generateData(e, t, a) {
    for (var n = 0, o = []; n < t;) {
      var r = Math.floor(Math.random() * (a.max - a.min + 1)) + a.min,
        i = Math.floor(61 * Math.random()) + 15;
      o.push([e, r, i]), e += 864e5, n++
    }
    return o
  }
  options = {
    chart: {
      height: 350,
      type: "bubble"
    },
    dataLabels: {
      enabled: !1
    },
    series: [{
      name: "Product1",
      data: generateData(new Date("11 Feb 2017 GMT").getTime(), 20, {
        min: 10,
        max: 40
      })
    }, {
      name: "Product2",
      data: generateData(new Date("11 Feb 2017 GMT").getTime(), 20, {
        min: 10,
        max: 40
      })
    }, {
      name: "Product3",
      data: generateData(new Date("11 Feb 2017 GMT").getTime(), 20, {
        min: 10,
        max: 40
      })
    }],
    fill: {
      type: "gradient"
    },
    colors: ["#4788ff", "#37e6b0", "#37e6b0"],
    title: {
      text: "3D Bubble Chart"
    },
    xaxis: {
      tickAmount: 12,
      type: "datetime",
      labels: {
        rotate: 0
      }
    },
    yaxis: {
      max: 40
    },
    theme: {
      palette: "palette2"
    }
  };
  (chart = new ApexCharts(document.querySelector("#apex-bubble-chart"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#apex-scatter-chart").length) {
  options = {
    chart: {
      height: 350,
      type: "scatter",
      zoom: {
        enabled: !0,
        type: "xy"
      }
    },
    colors: ["#4788ff", "#ff4b4b", "#37e6b0"],
    series: [{
      name: "SAMPLE A",
      data: [
        [16.4, 5.4],
        [21.7, 2],
        [10.9, 0],
        [10.9, 8.2],
        [16.4, 0],
        [16.4, 1.8],
        [13.6, .3],
        [13.6, 0],
        [29.9, 0],
        [27.1, 2.3],
        [16.4, 0],
        [13.6, 3.7],
        [10.9, 5.2],
        [16.4, 6.5],
        [10.9, 0],
        [24.5, 7.1],
        [10.9, 0],
        [8.1, 4.7]
      ]
    }, {
      name: "SAMPLE B",
      data: [
        [36.4, 13.4],
        [1.7, 11],
        [1.9, 9],
        [1.9, 13.2],
        [1.4, 7],
        [6.4, 8.8],
        [3.6, 4.3],
        [1.6, 10],
        [9.9, 2],
        [7.1, 15],
        [1.4, 0],
        [3.6, 13.7],
        [1.9, 15.2],
        [6.4, 16.5],
        [.9, 10],
        [4.5, 17.1],
        [10.9, 10],
        [.1, 14.7]
      ]
    }, {
      name: "SAMPLE C",
      data: [
        [21.7, 3],
        [23.6, 3.5],
        [28, 4],
        [27.1, .3],
        [16.4, 4],
        [13.6, 0],
        [19, 5],
        [22.4, 3],
        [24.5, 3],
        [32.6, 3],
        [27.1, 4],
        [29.6, 6],
        [31.6, 8],
        [21.6, 5],
        [20.9, 4],
        [22.4, 0],
        [32.6, 10.3],
        [29.7, 20.8]
      ]
    }],
    xaxis: {
      tickAmount: 5,
      labels: {
        formatter: function(e) {
          return parseFloat(e).toFixed(1)
        }
      }
    },
    yaxis: {
      tickAmount: 5
    }
  };
  (chart = new ApexCharts(document.querySelector("#apex-scatter-chart"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#apex-radialbar-chart").length) {
  options = {
    chart: {
      height: 350,
      type: "radialBar"
    },
    plotOptions: {
      radialBar: {
        dataLabels: {
          name: {
            fontSize: "22px"
          },
          value: {
            fontSize: "16px"
          },
          total: {
            show: !0,
            label: "Total",
            formatter: function(e) {
              return 249
            }
          }
        }
      }
    },
    series: [44, 55, 67, 83],
    labels: ["Apples", "Oranges", "Bananas", "Berries"],
    colors: ["#4788ff", "#ff4b4b", "#876cfe", "#37e6b0"]
  };
  (chart = new ApexCharts(document.querySelector("#apex-radialbar-chart"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#apex-pie-chart").length) {
  options = {
    chart: {
      width: 380,
      type: "pie"
    },
    labels: ["Team A", "Team B", "Team C", "Team D", "Team E"],
    series: [44, 55, 13, 43, 22],
    colors: ["#4788ff", "#ff4b4b", "#876cfe", "#37e6b0", "#c8c8c8"],
    responsive: [{
      breakpoint: 480,
      options: {
        chart: {
          width: 200
        },
        legend: {
          position: "bottom"
        }
      }
    }]
  };
  (chart = new ApexCharts(document.querySelector("#apex-pie-chart"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#advanced-chart").length) {
  var options = {
    series: [
    {
      name: 'Bob',
      data: [
        {
          x: 'Design',
          y: [
            new Date('2019-03-05').getTime(),
            new Date('2019-03-08').getTime()
          ]
        },
        {
          x: 'Code',
          y: [
            new Date('2019-03-02').getTime(),
            new Date('2019-03-05').getTime()
          ]
        },
        {
          x: 'Code',
          y: [
            new Date('2019-03-05').getTime(),
            new Date('2019-03-07').getTime()
          ]
        },
        {
          x: 'Test',
          y: [
            new Date('2019-03-03').getTime(),
            new Date('2019-03-09').getTime()
          ]
        },
        {
          x: 'Test',
          y: [
            new Date('2019-03-08').getTime(),
            new Date('2019-03-11').getTime()
          ]
        },
        {
          x: 'Validation',
          y: [
            new Date('2019-03-11').getTime(),
            new Date('2019-03-16').getTime()
          ]
        },
        {
          x: 'Design',
          y: [
            new Date('2019-03-01').getTime(),
            new Date('2019-03-03').getTime()
          ]
        }
      ]
    },
    {
      name: 'Joe',
      data: [
        {
          x: 'Design',
          y: [
            new Date('2019-03-02').getTime(),
            new Date('2019-03-05').getTime()
          ]
        },
        {
          x: 'Test',
          y: [
            new Date('2019-03-06').getTime(),
            new Date('2019-03-16').getTime()
          ]
        },
        {
          x: 'Code',
          y: [
            new Date('2019-03-03').getTime(),
            new Date('2019-03-07').getTime()
          ]
        },
        {
          x: 'Deployment',
          y: [
            new Date('2019-03-20').getTime(),
            new Date('2019-03-22').getTime()
          ]
        },
        {
          x: 'Design',
          y: [
            new Date('2019-03-10').getTime(),
            new Date('2019-03-16').getTime()
          ]
        }
      ]
    },
    {
      name: 'Dan',
      data: [
        {
          x: 'Code',
          y: [
            new Date('2019-03-10').getTime(),
            new Date('2019-03-17').getTime()
          ]
        },
        {
          x: 'Validation',
          y: [
            new Date('2019-03-05').getTime(),
            new Date('2019-03-09').getTime()
          ]
        },
      ]
    }
  ],
    colors: ["#4788ff", "#ff4b4b", "#37e6b0"],
    chart: {
    height: 450,
    type: 'rangeBar'
  },
  plotOptions: {
    bar: {
      horizontal: true,
      barHeight: '80%'
    }
  },
  xaxis: {
    type: 'datetime'
  },
  stroke: {
    width: 1
  },
  fill: {
    type: 'solid',
    opacity: 0.6
  },
  legend: {
    position: 'top',
    horizontalAlign: 'left'
  }
  };

  (chart = new ApexCharts(document.querySelector("#advanced-chart"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#radar-multiple-chart").length) {
  var options = {
    series: [{
    name: 'Series 1',
    data: [80, 50, 30, 40, 100, 20],
  }, {
    name: 'Series 2',
    data: [20, 30, 40, 80, 20, 80],
  }, {
    name: 'Series 3',
    data: [44, 76, 78, 13, 43, 10],
  }],
  colors: ["#4788ff", "#ff4b4b", "#37e6b0"],
    chart: {
    height: 350,
    type: 'radar',
    dropShadow: {
      enabled: true,
      blur: 1,
      left: 1,
      top: 1
    }
  },
  title: {
    text: 'Radar Chart - Multi Series'
  },
  stroke: {
    width: 2
  },
  fill: {
    opacity: 0.1
  },
  markers: {
    size: 0
  },
  xaxis: {
    categories: ['2011', '2012', '2013', '2014', '2015', '2016']
  }
  };

  (chart = new ApexCharts(document.querySelector("#radar-multiple-chart"), options)).render()
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}



/*---------------------------------------------------------------------
   Am Charts
-----------------------------------------------------------------------*/

    if(jQuery('#am-simple-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      // Create chart instance
      var chart = am4core.create("am-simple-chart", am4charts.XYChart);
      chart.colors.list = [am4core.color("#4788ff"),];

      // Add data
      chart.data = [{
        "country": "USA",
        "visits": 2025
      }, {
        "country": "China",
        "visits": 1882
      }, {
        "country": "Japan",
        "visits": 1809
      }, {
        "country": "Germany",
        "visits": 1322
      }, {
        "country": "UK",
        "visits": 1122
      }, {
        "country": "France",
        "visits": 1114
      }];

      // Create axes

      var categoryAxis = chart.xAxes.push(new am4charts.CategoryAxis());
      categoryAxis.dataFields.category = "country";
      categoryAxis.renderer.grid.template.location = 0;
      categoryAxis.renderer.minGridDistance = 30;

      categoryAxis.renderer.labels.template.adapter.add("dy", function(dy, target) {
        if (target.dataItem && target.dataItem.index & 2 == 2) {
          return dy + 25;
        }
        return dy;
      });

      var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());

      // Create series
      var series = chart.series.push(new am4charts.ColumnSeries());
      series.dataFields.valueY = "visits";
      series.dataFields.categoryX = "country";
      series.name = "Visits";
      series.columns.template.tooltipText = "{categoryX}: [bold]{valueY}[/]";
      series.columns.template.fillOpacity = .8;

      var columnTemplate = series.columns.template;
      columnTemplate.strokeWidth = 2;
      columnTemplate.strokeOpacity = 1;

      const body = document.querySelector('body')
      if (body.classList.contains('dark')) {
        amChartUpdate(chart, {
          dark: true
        })
      }

      document.addEventListener('ChangeColorMode', function (e) {
        amChartUpdate(chart, e.detail)
      })

      }); // end am4core.ready()
   }

   if(jQuery('#am-columnlinr-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      // Create chart instance
      var chart = am4core.create("am-columnlinr-chart", am4charts.XYChart);
       chart.colors.list = [am4core.color("#4788ff"),];

      // Export
      chart.exporting.menu = new am4core.ExportMenu();

      // Data for both series
      var data = [ {
        "year": "2009",
        "income": 23.5,
        "expenses": 21.1
      }, {
        "year": "2010",
        "income": 26.2,
        "expenses": 30.5
      }, {
        "year": "2011",
        "income": 30.1,
        "expenses": 34.9
      }, {
        "year": "2012",
        "income": 29.5,
        "expenses": 31.1
      }, {
        "year": "2013",
        "income": 30.6,
        "expenses": 28.2,
        "lineDash": "5,5",
      }, {
        "year": "2014",
        "income": 34.1,
        "expenses": 32.9,
        "strokeWidth": 1,
        "columnDash": "5,5",
        "fillOpacity": 0.2,
        "additional": "(projection)"
      } ];

      /* Create axes */
      var categoryAxis = chart.xAxes.push(new am4charts.CategoryAxis());
      categoryAxis.dataFields.category = "year";
      categoryAxis.renderer.minGridDistance = 30;

      /* Create value axis */
      var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());

      /* Create series */
      var columnSeries = chart.series.push(new am4charts.ColumnSeries());
      columnSeries.name = "Income";
      columnSeries.dataFields.valueY = "income";
      columnSeries.dataFields.categoryX = "year";

      columnSeries.columns.template.tooltipText = "[#fff font-size: 15px]{name} in {categoryX}:\n[/][#fff font-size: 20px]{valueY}[/] [#fff]{additional}[/]"
      columnSeries.columns.template.propertyFields.fillOpacity = "fillOpacity";
      columnSeries.columns.template.propertyFields.stroke = "stroke";
      columnSeries.columns.template.propertyFields.strokeWidth = "strokeWidth";
      columnSeries.columns.template.propertyFields.strokeDasharray = "columnDash";
      columnSeries.tooltip.label.textAlign = "middle";

      var lineSeries = chart.series.push(new am4charts.LineSeries());
      lineSeries.name = "Expenses";
      lineSeries.dataFields.valueY = "expenses";
      lineSeries.dataFields.categoryX = "year";

      lineSeries.stroke = am4core.color("#4788ff");
      lineSeries.strokeWidth = 3;
      lineSeries.propertyFields.strokeDasharray = "lineDash";
      lineSeries.tooltip.label.textAlign = "middle";

      var bullet = lineSeries.bullets.push(new am4charts.Bullet());
      bullet.fill = am4core.color("#fdd400"); // tooltips grab fill from parent by default
      bullet.tooltipText = "[#fff font-size: 15px]{name} in {categoryX}:\n[/][#fff font-size: 20px]{valueY}[/] [#fff]{additional}[/]"
      var circle = bullet.createChild(am4core.Circle);
      circle.radius = 4;
      circle.fill = am4core.color("#fff");
      circle.strokeWidth = 3;

      chart.data = data;

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      });
   }

   if(jQuery('#am-stackedcolumn-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      // Create chart instance
      var chart = am4core.create("am-stackedcolumn-chart", am4charts.XYChart);
      chart.colors.list = [am4core.color("#ff4b4b"),
      am4core.color("#37e6b0"),
      am4core.color("#fe721c")];


      // Add data
      chart.data = [{
        "year": "2016",
        "europe": 2.5,
        "namerica": 2.5,
        "asia": 2.1,
        "lamerica": 0.3,
        "meast": 0.2
      }, {
        "year": "2017",
        "europe": 2.6,
        "namerica": 2.7,
        "asia": 2.2,
        "lamerica": 0.3,
        "meast": 0.3
      }, {
        "year": "2018",
        "europe": 2.8,
        "namerica": 2.9,
        "asia": 2.4,
        "lamerica": 0.3,
        "meast": 0.3
      }];

      // Create axes
      var categoryAxis = chart.xAxes.push(new am4charts.CategoryAxis());
      categoryAxis.dataFields.category = "year";
      categoryAxis.renderer.grid.template.location = 0;


      var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());
      valueAxis.renderer.inside = true;
      valueAxis.renderer.labels.template.disabled = true;
      valueAxis.min = 0;

      // Create series
      function createSeries(field, name) {

        // Set up series
        var series = chart.series.push(new am4charts.ColumnSeries());
        series.name = name;
        series.dataFields.valueY = field;
        series.dataFields.categoryX = "year";
        series.sequencedInterpolation = true;

        // Make it stacked
        series.stacked = true;

        // Configure columns
        series.columns.template.width = am4core.percent(60);
        series.columns.template.tooltipText = "[bold]{name}[/]\n[font-size:14px]{categoryX}: {valueY}";

        // Add label
        var labelBullet = series.bullets.push(new am4charts.LabelBullet());
        labelBullet.label.text = "{valueY}";
        labelBullet.locationY = 0.5;

        return series;
      }

      createSeries("europe", "Europe");
      createSeries("namerica", "North America");
      createSeries("asia", "Asia-Pacific");

      // Legend
      chart.legend = new am4charts.Legend();

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      }); // end am4core.ready()
   }

   if(jQuery('#am-barline-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      var chart = am4core.create("am-barline-chart", am4charts.XYChart);
      chart.colors.list = [am4core.color("#4788ff"),
      am4core.color("#37e6b0")];

      chart.data = [{
              "year": "2005",
              "income": 23.5,
              "expenses": 18.1
          }, {
              "year": "2006",
              "income": 26.2,
              "expenses": 22.8
          }, {
              "year": "2007",
              "income": 30.1,
              "expenses": 23.9
          }, {
              "year": "2008",
              "income": 29.5,
              "expenses": 25.1
          }, {
              "year": "2009",
              "income": 24.6,
              "expenses": 25
          }];

      //create category axis for years
      var categoryAxis = chart.yAxes.push(new am4charts.CategoryAxis());
      categoryAxis.dataFields.category = "year";
      categoryAxis.renderer.inversed = true;
      categoryAxis.renderer.grid.template.location = 0;

      //create value axis for income and expenses
      var valueAxis = chart.xAxes.push(new am4charts.ValueAxis());
      valueAxis.renderer.opposite = true;


      //create columns
      var series = chart.series.push(new am4charts.ColumnSeries());
      series.dataFields.categoryY = "year";
      series.dataFields.valueX = "income";
      series.name = "Income";
      series.columns.template.fillOpacity = 0.5;
      series.columns.template.strokeOpacity = 0;
      series.tooltipText = "Income in {categoryY}: {valueX.value}";

      //create line
      var lineSeries = chart.series.push(new am4charts.LineSeries());
      lineSeries.dataFields.categoryY = "year";
      lineSeries.dataFields.valueX = "expenses";
      lineSeries.name = "Expenses";
      lineSeries.strokeWidth = 3;
      lineSeries.tooltipText = "Expenses in {categoryY}: {valueX.value}";

      //add bullets
      var circleBullet = lineSeries.bullets.push(new am4charts.CircleBullet());
      circleBullet.circle.fill = am4core.color("#fff");
      circleBullet.circle.strokeWidth = 2;

      //add chart cursor
      chart.cursor = new am4charts.XYCursor();
      chart.cursor.behavior = "zoomY";

      //add legend
      chart.legend = new am4charts.Legend();

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      }); // end am4core.ready()
   }

   if(jQuery('#am-datedata-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      // Create chart instance
      var chart = am4core.create("am-datedata-chart", am4charts.XYChart);
      chart.colors.list = [am4core.color("#57aeff")];

      // Add data
      chart.data = [{
        "date": "2012-07-27",
        "value": 13
      }, {
        "date": "2012-07-28",
        "value": 11
      }, {
        "date": "2012-07-29",
        "value": 15
      }, {
        "date": "2012-07-30",
        "value": 16
      }, {
        "date": "2012-07-31",
        "value": 18
      }, {
        "date": "2012-08-01",
        "value": 13
      }, {
        "date": "2012-08-02",
        "value": 22
      }, {
        "date": "2012-08-03",
        "value": 23
      }, {
        "date": "2012-08-04",
        "value": 20
      }, {
        "date": "2012-08-05",
        "value": 17
      }, {
        "date": "2012-08-06",
        "value": 16
      }, {
        "date": "2012-08-07",
        "value": 18
      }, {
        "date": "2012-08-08",
        "value": 21
      }, {
        "date": "2012-08-09",
        "value": 26
      }, {
        "date": "2012-08-10",
        "value": 24
      }, {
        "date": "2012-08-11",
        "value": 29
      }, {
        "date": "2012-08-12",
        "value": 32
      }, {
        "date": "2012-08-13",
        "value": 18
      }, {
        "date": "2012-08-14",
        "value": 24
      }, {
        "date": "2012-08-15",
        "value": 22
      }, {
        "date": "2012-08-16",
        "value": 18
      }, {
        "date": "2012-08-17",
        "value": 19
      }, {
        "date": "2012-08-18",
        "value": 14
      }, {
        "date": "2012-08-19",
        "value": 15
      }, {
        "date": "2012-08-20",
        "value": 12
      }, {
        "date": "2012-08-21",
        "value": 8
      }, {
        "date": "2012-08-22",
        "value": 9
      }, {
        "date": "2012-08-23",
        "value": 8
      }, {
        "date": "2012-08-24",
        "value": 7
      }, {
        "date": "2012-08-25",
        "value": 5
      }, {
        "date": "2012-08-26",
        "value": 11
      }, {
        "date": "2012-08-27",
        "value": 13
      }, {
        "date": "2012-08-28",
        "value": 18
      }, {
        "date": "2012-08-29",
        "value": 20
      }, {
        "date": "2012-08-30",
        "value": 29
      }, {
        "date": "2012-08-31",
        "value": 33
      }, {
        "date": "2012-09-01",
        "value": 42
      }, {
        "date": "2012-09-02",
        "value": 35
      }, {
        "date": "2012-09-03",
        "value": 31
      }, {
        "date": "2012-09-04",
        "value": 47
      }, {
        "date": "2012-09-05",
        "value": 52
      }, {
        "date": "2012-09-06",
        "value": 46
      }, {
        "date": "2012-09-07",
        "value": 41
      }, {
        "date": "2012-09-08",
        "value": 43
      }, {
        "date": "2012-09-09",
        "value": 40
      }, {
        "date": "2012-09-10",
        "value": 39
      }, {
        "date": "2012-09-11",
        "value": 34
      }, {
        "date": "2012-09-12",
        "value": 29
      }, {
        "date": "2012-09-13",
        "value": 34
      }, {
        "date": "2012-09-14",
        "value": 37
      }, {
        "date": "2012-09-15",
        "value": 42
      }, {
        "date": "2012-09-16",
        "value": 49
      }, {
        "date": "2012-09-17",
        "value": 46
      }, {
        "date": "2012-09-18",
        "value": 47
      }, {
        "date": "2012-09-19",
        "value": 55
      }, {
        "date": "2012-09-20",
        "value": 59
      }, {
        "date": "2012-09-21",
        "value": 58
      }, {
        "date": "2012-09-22",
        "value": 57
      }, {
        "date": "2012-09-23",
        "value": 61
      }, {
        "date": "2012-09-24",
        "value": 59
      }, {
        "date": "2012-09-25",
        "value": 67
      }, {
        "date": "2012-09-26",
        "value": 65
      }, {
        "date": "2012-09-27",
        "value": 61
      }, {
        "date": "2012-09-28",
        "value": 66
      }, {
        "date": "2012-09-29",
        "value": 69
      }, {
        "date": "2012-09-30",
        "value": 71
      }, {
        "date": "2012-10-01",
        "value": 67
      }, {
        "date": "2012-10-02",
        "value": 63
      }, {
        "date": "2012-10-03",
        "value": 46
      }, {
        "date": "2012-10-04",
        "value": 32
      }, {
        "date": "2012-10-05",
        "value": 21
      }, {
        "date": "2012-10-06",
        "value": 18
      }, {
        "date": "2012-10-07",
        "value": 21
      }, {
        "date": "2012-10-08",
        "value": 28
      }, {
        "date": "2012-10-09",
        "value": 27
      }, {
        "date": "2012-10-10",
        "value": 36
      }, {
        "date": "2012-10-11",
        "value": 33
      }, {
        "date": "2012-10-12",
        "value": 31
      }, {
        "date": "2012-10-13",
        "value": 30
      }, {
        "date": "2012-10-14",
        "value": 34
      }, {
        "date": "2012-10-15",
        "value": 38
      }, {
        "date": "2012-10-16",
        "value": 37
      }, {
        "date": "2012-10-17",
        "value": 44
      }, {
        "date": "2012-10-18",
        "value": 49
      }, {
        "date": "2012-10-19",
        "value": 53
      }, {
        "date": "2012-10-20",
        "value": 57
      }, {
        "date": "2012-10-21",
        "value": 60
      }, {
        "date": "2012-10-22",
        "value": 61
      }, {
        "date": "2012-10-23",
        "value": 69
      }, {
        "date": "2012-10-24",
        "value": 67
      }, {
        "date": "2012-10-25",
        "value": 72
      }, {
        "date": "2012-10-26",
        "value": 77
      }, {
        "date": "2012-10-27",
        "value": 75
      }, {
        "date": "2012-10-28",
        "value": 70
      }, {
        "date": "2012-10-29",
        "value": 72
      }, {
        "date": "2012-10-30",
        "value": 70
      }, {
        "date": "2012-10-31",
        "value": 72
      }, {
        "date": "2012-11-01",
        "value": 73
      }, {
        "date": "2012-11-02",
        "value": 67
      }, {
        "date": "2012-11-03",
        "value": 68
      }, {
        "date": "2012-11-04",
        "value": 65
      }, {
        "date": "2012-11-05",
        "value": 71
      }, {
        "date": "2012-11-06",
        "value": 75
      }, {
        "date": "2012-11-07",
        "value": 74
      }, {
        "date": "2012-11-08",
        "value": 71
      }, {
        "date": "2012-11-09",
        "value": 76
      }, {
        "date": "2012-11-10",
        "value": 77
      }, {
        "date": "2012-11-11",
        "value": 81
      }, {
        "date": "2012-11-12",
        "value": 83
      }, {
        "date": "2012-11-13",
        "value": 80
      }, {
        "date": "2012-11-14",
        "value": 81
      }, {
        "date": "2012-11-15",
        "value": 87
      }, {
        "date": "2012-11-16",
        "value": 82
      }, {
        "date": "2012-11-17",
        "value": 86
      }, {
        "date": "2012-11-18",
        "value": 80
      }, {
        "date": "2012-11-19",
        "value": 87
      }, {
        "date": "2012-11-20",
        "value": 83
      }, {
        "date": "2012-11-21",
        "value": 85
      }, {
        "date": "2012-11-22",
        "value": 84
      }, {
        "date": "2012-11-23",
        "value": 82
      }, {
        "date": "2012-11-24",
        "value": 73
      }, {
        "date": "2012-11-25",
        "value": 71
      }, {
        "date": "2012-11-26",
        "value": 75
      }, {
        "date": "2012-11-27",
        "value": 79
      }, {
        "date": "2012-11-28",
        "value": 70
      }, {
        "date": "2012-11-29",
        "value": 73
      }, {
        "date": "2012-11-30",
        "value": 61
      }, {
        "date": "2012-12-01",
        "value": 62
      }, {
        "date": "2012-12-02",
        "value": 66
      }, {
        "date": "2012-12-03",
        "value": 65
      }, {
        "date": "2012-12-04",
        "value": 73
      }, {
        "date": "2012-12-05",
        "value": 79
      }, {
        "date": "2012-12-06",
        "value": 78
      }, {
        "date": "2012-12-07",
        "value": 78
      }, {
        "date": "2012-12-08",
        "value": 78
      }, {
        "date": "2012-12-09",
        "value": 74
      }, {
        "date": "2012-12-10",
        "value": 73
      }, {
        "date": "2012-12-11",
        "value": 75
      }, {
        "date": "2012-12-12",
        "value": 70
      }, {
        "date": "2012-12-13",
        "value": 77
      }, {
        "date": "2012-12-14",
        "value": 67
      }, {
        "date": "2012-12-15",
        "value": 62
      }, {
        "date": "2012-12-16",
        "value": 64
      }, {
        "date": "2012-12-17",
        "value": 61
      }, {
        "date": "2012-12-18",
        "value": 59
      }, {
        "date": "2012-12-19",
        "value": 53
      }, {
        "date": "2012-12-20",
        "value": 54
      }, {
        "date": "2012-12-21",
        "value": 56
      }, {
        "date": "2012-12-22",
        "value": 59
      }, {
        "date": "2012-12-23",
        "value": 58
      }, {
        "date": "2012-12-24",
        "value": 55
      }, {
        "date": "2012-12-25",
        "value": 52
      }, {
        "date": "2012-12-26",
        "value": 54
      }, {
        "date": "2012-12-27",
        "value": 50
      }, {
        "date": "2012-12-28",
        "value": 50
      }, {
        "date": "2012-12-29",
        "value": 51
      }, {
        "date": "2012-12-30",
        "value": 52
      }, {
        "date": "2012-12-31",
        "value": 58
      }, {
        "date": "2013-01-01",
        "value": 60
      }, {
        "date": "2013-01-02",
        "value": 67
      }, {
        "date": "2013-01-03",
        "value": 64
      }, {
        "date": "2013-01-04",
        "value": 66
      }, {
        "date": "2013-01-05",
        "value": 60
      }, {
        "date": "2013-01-06",
        "value": 63
      }, {
        "date": "2013-01-07",
        "value": 61
      }, {
        "date": "2013-01-08",
        "value": 60
      }, {
        "date": "2013-01-09",
        "value": 65
      }, {
        "date": "2013-01-10",
        "value": 75
      }, {
        "date": "2013-01-11",
        "value": 77
      }, {
        "date": "2013-01-12",
        "value": 78
      }, {
        "date": "2013-01-13",
        "value": 70
      }, {
        "date": "2013-01-14",
        "value": 70
      }, {
        "date": "2013-01-15",
        "value": 73
      }, {
        "date": "2013-01-16",
        "value": 71
      }, {
        "date": "2013-01-17",
        "value": 74
      }, {
        "date": "2013-01-18",
        "value": 78
      }, {
        "date": "2013-01-19",
        "value": 85
      }, {
        "date": "2013-01-20",
        "value": 82
      }, {
        "date": "2013-01-21",
        "value": 83
      }, {
        "date": "2013-01-22",
        "value": 88
      }, {
        "date": "2013-01-23",
        "value": 85
      }, {
        "date": "2013-01-24",
        "value": 85
      }, {
        "date": "2013-01-25",
        "value": 80
      }, {
        "date": "2013-01-26",
        "value": 87
      }, {
        "date": "2013-01-27",
        "value": 84
      }, {
        "date": "2013-01-28",
        "value": 83
      }, {
        "date": "2013-01-29",
        "value": 84
      }, {
        "date": "2013-01-30",
        "value": 81
      }];

      // Set input format for the dates
      chart.dateFormatter.inputDateFormat = "yyyy-MM-dd";

      // Create axes
      var dateAxis = chart.xAxes.push(new am4charts.DateAxis());
      var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());

      // Create series
      var series = chart.series.push(new am4charts.LineSeries());
      series.dataFields.valueY = "value";
      series.dataFields.dateX = "date";
      series.tooltipText = "{value}"
      series.strokeWidth = 2;
      series.minBulletDistance = 15;

      // Drop-shaped tooltips
      series.tooltip.background.cornerRadius = 20;
      series.tooltip.background.strokeOpacity = 0;
      series.tooltip.pointerOrientation = "vertical";
      series.tooltip.label.minWidth = 40;
      series.tooltip.label.minHeight = 40;
      series.tooltip.label.textAlign = "middle";
      series.tooltip.label.textValign = "middle";

      // Make bullets grow on hover
      var bullet = series.bullets.push(new am4charts.CircleBullet());
      bullet.circle.strokeWidth = 2;
      bullet.circle.radius = 4;
      bullet.circle.fill = am4core.color("#fff");

      var bullethover = bullet.states.create("hover");
      bullethover.properties.scale = 1.3;

      // Make a panning cursor
      chart.cursor = new am4charts.XYCursor();
      chart.cursor.behavior = "panXY";
      chart.cursor.xAxis = dateAxis;
      chart.cursor.snapToSeries = series;

      // Create vertical scrollbar and place it before the value axis
      chart.scrollbarY = new am4core.Scrollbar();
      chart.scrollbarY.parent = chart.leftAxesContainer;
      chart.scrollbarY.toBack();

      // Create a horizontal scrollbar with previe and place it underneath the date axis
      chart.scrollbarX = new am4charts.XYChartScrollbar();
      chart.scrollbarX.series.push(series);
      chart.scrollbarX.parent = chart.bottomAxesContainer;

      dateAxis.start = 0.79;
      dateAxis.keepSelection = true;

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })


      }); // end am4core.ready()
   }
   if(jQuery('#am-linescrollzoiq-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      // Create chart instance
      var chart = am4core.create("am-linescrollzoiq-chart", am4charts.XYChart);
      chart.colors.list = [am4core.color("#57aeff")];

      // Add data
      chart.data = generateChartData();

      // Create axes
      var dateAxis = chart.xAxes.push(new am4charts.DateAxis());
      dateAxis.renderer.minGridDistance = 50;

      var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());

      // Create series
      var series = chart.series.push(new am4charts.LineSeries());
      series.dataFields.valueY = "visits";
      series.dataFields.dateX = "date";
      series.strokeWidth = 2;
      series.minBulletDistance = 10;
      series.tooltipText = "{valueY}";
      series.tooltip.pointerOrientation = "vertical";
      series.tooltip.background.cornerRadius = 20;
      series.tooltip.background.fillOpacity = 0.5;
      series.tooltip.label.padding(12,12,12,12)

      // Add scrollbar
      chart.scrollbarX = new am4charts.XYChartScrollbar();
      chart.scrollbarX.series.push(series);

      // Add cursor
      chart.cursor = new am4charts.XYCursor();
      chart.cursor.xAxis = dateAxis;
      chart.cursor.snapToSeries = series;

      function generateChartData() {
          var chartData = [];
          var firstDate = new Date();
          firstDate.setDate(firstDate.getDate() - 1000);
          var visits = 1200;
          for (var i = 0; i < 500; i++) {
              var newDate = new Date(firstDate);
              newDate.setDate(newDate.getDate() + i);

              visits += Math.round((Math.random()<0.5?1:-1)*Math.random()*10);

              chartData.push({
                  date: newDate,
                  visits: visits
              });
          }
          return chartData;
      }

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      });
   }

   if(jQuery('#am-zoomable-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      // Create chart instance
      var chart = am4core.create("am-zoomable-chart", am4charts.XYChart);
      chart.colors.list = [am4core.color("#57aeff")];

      // Add data
      chart.data = [ {
        "date": "2012-07-27",
        "value": 13
      }, {
        "date": "2012-07-28",
        "value": 11
      }, {
        "date": "2012-07-29",
        "value": 15
      }, {
        "date": "2012-07-30",
        "value": 16
      }, {
        "date": "2012-07-31",
        "value": 18
      }, {
        "date": "2012-08-01",
        "value": 13
      }, {
        "date": "2012-08-02",
        "value": 22
      }, {
        "date": "2012-08-03",
        "value": 23
      }, {
        "date": "2012-08-04",
        "value": 20
      }, {
        "date": "2012-08-05",
        "value": 17
      }, {
        "date": "2012-08-06",
        "value": 16
      }, {
        "date": "2012-08-07",
        "value": 18
      }, {
        "date": "2012-08-08",
        "value": 21
      }, {
        "date": "2012-08-09",
        "value": 26
      }, {
        "date": "2012-08-10",
        "value": 24
      }, {
        "date": "2012-08-11",
        "value": 29
      }, {
        "date": "2012-08-12",
        "value": 32
      }, {
        "date": "2012-08-13",
        "value": 18
      }, {
        "date": "2012-08-14",
        "value": 24
      }, {
        "date": "2012-08-15",
        "value": 22
      }, {
        "date": "2012-08-16",
        "value": 18
      }, {
        "date": "2012-08-17",
        "value": 19
      }, {
        "date": "2012-08-18",
        "value": 14
      }, {
        "date": "2012-08-19",
        "value": 15
      }, {
        "date": "2012-08-20",
        "value": 12
      }, {
        "date": "2012-08-21",
        "value": 8
      }, {
        "date": "2012-08-22",
        "value": 9
      }, {
        "date": "2012-08-23",
        "value": 8
      }, {
        "date": "2012-08-24",
        "value": 7
      }, {
        "date": "2012-08-25",
        "value": 5
      }, {
        "date": "2012-08-26",
        "value": 11
      }, {
        "date": "2012-08-27",
        "value": 13
      }, {
        "date": "2012-08-28",
        "value": 18
      }, {
        "date": "2012-08-29",
        "value": 20
      }, {
        "date": "2012-08-30",
        "value": 29
      }, {
        "date": "2012-08-31",
        "value": 33
      }, {
        "date": "2012-09-01",
        "value": 42
      }, {
        "date": "2012-09-02",
        "value": 35
      }, {
        "date": "2012-09-03",
        "value": 31
      }, {
        "date": "2012-09-04",
        "value": 47
      }, {
        "date": "2012-09-05",
        "value": 52
      }, {
        "date": "2012-09-06",
        "value": 46
      }, {
        "date": "2012-09-07",
        "value": 41
      }, {
        "date": "2012-09-08",
        "value": 43
      }, {
        "date": "2012-09-09",
        "value": 40
      }, {
        "date": "2012-09-10",
        "value": 39
      }, {
        "date": "2012-09-11",
        "value": 34
      }, {
        "date": "2012-09-12",
        "value": 29
      }, {
        "date": "2012-09-13",
        "value": 34
      }, {
        "date": "2012-09-14",
        "value": 37
      }, {
        "date": "2012-09-15",
        "value": 42
      }, {
        "date": "2012-09-16",
        "value": 49
      }, {
        "date": "2012-09-17",
        "value": 46
      }, {
        "date": "2012-09-18",
        "value": 47
      }, {
        "date": "2012-09-19",
        "value": 55
      }, {
        "date": "2012-09-20",
        "value": 59
      }, {
        "date": "2012-09-21",
        "value": 58
      }, {
        "date": "2012-09-22",
        "value": 57
      }, {
        "date": "2012-09-23",
        "value": 61
      }, {
        "date": "2012-09-24",
        "value": 59
      }, {
        "date": "2012-09-25",
        "value": 67
      }, {
        "date": "2012-09-26",
        "value": 65
      }, {
        "date": "2012-09-27",
        "value": 61
      }, {
        "date": "2012-09-28",
        "value": 66
      }, {
        "date": "2012-09-29",
        "value": 69
      }, {
        "date": "2012-09-30",
        "value": 71
      }, {
        "date": "2012-10-01",
        "value": 67
      }, {
        "date": "2012-10-02",
        "value": 63
      }, {
        "date": "2012-10-03",
        "value": 46
      }, {
        "date": "2012-10-04",
        "value": 32
      }, {
        "date": "2012-10-05",
        "value": 21
      }, {
        "date": "2012-10-06",
        "value": 18
      }, {
        "date": "2012-10-07",
        "value": 21
      }, {
        "date": "2012-10-08",
        "value": 28
      }, {
        "date": "2012-10-09",
        "value": 27
      }, {
        "date": "2012-10-10",
        "value": 36
      }, {
        "date": "2012-10-11",
        "value": 33
      }, {
        "date": "2012-10-12",
        "value": 31
      }, {
        "date": "2012-10-13",
        "value": 30
      }, {
        "date": "2012-10-14",
        "value": 34
      }, {
        "date": "2012-10-15",
        "value": 38
      }, {
        "date": "2012-10-16",
        "value": 37
      }, {
        "date": "2012-10-17",
        "value": 44
      }, {
        "date": "2012-10-18",
        "value": 49
      }, {
        "date": "2012-10-19",
        "value": 53
      }, {
        "date": "2012-10-20",
        "value": 57
      }, {
        "date": "2012-10-21",
        "value": 60
      }, {
        "date": "2012-10-22",
        "value": 61
      }, {
        "date": "2012-10-23",
        "value": 69
      }, {
        "date": "2012-10-24",
        "value": 67
      }, {
        "date": "2012-10-25",
        "value": 72
      }, {
        "date": "2012-10-26",
        "value": 77
      }, {
        "date": "2012-10-27",
        "value": 75
      }, {
        "date": "2012-10-28",
        "value": 70
      }, {
        "date": "2012-10-29",
        "value": 72
      }, {
        "date": "2012-10-30",
        "value": 70
      }, {
        "date": "2012-10-31",
        "value": 72
      }, {
        "date": "2012-11-01",
        "value": 73
      }, {
        "date": "2012-11-02",
        "value": 67
      }, {
        "date": "2012-11-03",
        "value": 68
      }, {
        "date": "2012-11-04",
        "value": 65
      }, {
        "date": "2012-11-05",
        "value": 71
      }, {
        "date": "2012-11-06",
        "value": 75
      }, {
        "date": "2012-11-07",
        "value": 74
      }, {
        "date": "2012-11-08",
        "value": 71
      }, {
        "date": "2012-11-09",
        "value": 76
      }, {
        "date": "2012-11-10",
        "value": 77
      }, {
        "date": "2012-11-11",
        "value": 81
      }, {
        "date": "2012-11-12",
        "value": 83
      }, {
        "date": "2012-11-13",
        "value": 80
      }, {
        "date": "2012-11-18",
        "value": 80
      }, {
        "date": "2012-11-19",
        "value": 87
      }, {
        "date": "2012-11-20",
        "value": 83
      }, {
        "date": "2012-11-21",
        "value": 85
      }, {
        "date": "2012-11-22",
        "value": 84
      }, {
        "date": "2012-11-23",
        "value": 82
      }, {
        "date": "2012-11-24",
        "value": 73
      }, {
        "date": "2012-11-25",
        "value": 71
      }, {
        "date": "2012-11-26",
        "value": 75
      }, {
        "date": "2012-11-27",
        "value": 79
      }, {
        "date": "2012-11-28",
        "value": 70
      }, {
        "date": "2012-11-29",
        "value": 73
      }, {
        "date": "2012-11-30",
        "value": 61
      }, {
        "date": "2012-12-01",
        "value": 62
      }, {
        "date": "2012-12-02",
        "value": 66
      }, {
        "date": "2012-12-03",
        "value": 65
      }, {
        "date": "2012-12-04",
        "value": 73
      }, {
        "date": "2012-12-05",
        "value": 79
      }, {
        "date": "2012-12-06",
        "value": 78
      }, {
        "date": "2012-12-07",
        "value": 78
      }, {
        "date": "2012-12-08",
        "value": 78
      }, {
        "date": "2012-12-09",
        "value": 74
      }, {
        "date": "2012-12-10",
        "value": 73
      }, {
        "date": "2012-12-11",
        "value": 75
      }, {
        "date": "2012-12-12",
        "value": 70
      }, {
        "date": "2012-12-13",
        "value": 77
      }, {
        "date": "2012-12-14",
        "value": 67
      }, {
        "date": "2012-12-15",
        "value": 62
      }, {
        "date": "2012-12-16",
        "value": 64
      }, {
        "date": "2012-12-17",
        "value": 61
      }, {
        "date": "2012-12-18",
        "value": 59
      }, {
        "date": "2012-12-19",
        "value": 53
      }, {
        "date": "2012-12-20",
        "value": 54
      }, {
        "date": "2012-12-21",
        "value": 56
      }, {
        "date": "2012-12-22",
        "value": 59
      }, {
        "date": "2012-12-23",
        "value": 58
      }, {
        "date": "2012-12-24",
        "value": 55
      }, {
        "date": "2012-12-25",
        "value": 52
      }, {
        "date": "2012-12-26",
        "value": 54
      }, {
        "date": "2012-12-27",
        "value": 50
      }, {
        "date": "2012-12-28",
        "value": 50
      }, {
        "date": "2012-12-29",
        "value": 51
      }, {
        "date": "2012-12-30",
        "value": 52
      }, {
        "date": "2012-12-31",
        "value": 58
      }, {
        "date": "2013-01-01",
        "value": 60
      }, {
        "date": "2013-01-02",
        "value": 67
      }, {
        "date": "2013-01-03",
        "value": 64
      }, {
        "date": "2013-01-04",
        "value": 66
      }, {
        "date": "2013-01-05",
        "value": 60
      }, {
        "date": "2013-01-06",
        "value": 63
      }, {
        "date": "2013-01-07",
        "value": 61
      }, {
        "date": "2013-01-08",
        "value": 60
      }, {
        "date": "2013-01-09",
        "value": 65
      }, {
        "date": "2013-01-10",
        "value": 75
      }, {
        "date": "2013-01-11",
        "value": 77
      }, {
        "date": "2013-01-12",
        "value": 78
      }, {
        "date": "2013-01-13",
        "value": 70
      }, {
        "date": "2013-01-14",
        "value": 70
      }, {
        "date": "2013-01-15",
        "value": 73
      }, {
        "date": "2013-01-16",
        "value": 71
      }, {
        "date": "2013-01-17",
        "value": 74
      }, {
        "date": "2013-01-18",
        "value": 78
      }, {
        "date": "2013-01-19",
        "value": 85
      }, {
        "date": "2013-01-20",
        "value": 82
      }, {
        "date": "2013-01-21",
        "value": 83
      }, {
        "date": "2013-01-22",
        "value": 88
      }, {
        "date": "2013-01-23",
        "value": 85
      }, {
        "date": "2013-01-24",
        "value": 85
      }, {
        "date": "2013-01-25",
        "value": 80
      }, {
        "date": "2013-01-26",
        "value": 87
      }, {
        "date": "2013-01-27",
        "value": 84
      }, {
        "date": "2013-01-28",
        "value": 83
      }, {
        "date": "2013-01-29",
        "value": 84
      }, {
        "date": "2013-01-30",
        "value": 81
      } ];

      // Create axes
      var dateAxis = chart.xAxes.push(new am4charts.DateAxis());
      dateAxis.renderer.grid.template.location = 0;
      dateAxis.renderer.minGridDistance = 50;

      var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());

      // Create series
      var series = chart.series.push(new am4charts.LineSeries());
      series.dataFields.valueY = "value";
      series.dataFields.dateX = "date";
      series.strokeWidth = 3;
      series.fillOpacity = 0.5;

      // Add vertical scrollbar
      chart.scrollbarY = new am4core.Scrollbar();
      chart.scrollbarY.marginLeft = 0;

      // Add cursor
      chart.cursor = new am4charts.XYCursor();
      chart.cursor.behavior = "zoomY";
      chart.cursor.lineX.disabled = true;

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      }); // end am4core.ready()
   }
   if(jQuery('#am-radar-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      /* Create chart instance */
      var chart = am4core.create("am-radar-chart", am4charts.RadarChart);
      chart.colors.list = [am4core.color("#4788ff")];

      /* Add data */
      chart.data = [{
        "country": "Lithuania",
        "litres": 501
      }, {
        "country": "Czechia",
        "litres": 301
      }, {
        "country": "Ireland",
        "litres": 266
      }, {
        "country": "Germany",
        "litres": 165
      }, {
        "country": "Australia",
        "litres": 139
      }, {
        "country": "Austria",
        "litres": 336
      }, {
        "country": "UK",
        "litres": 290
      }, {
        "country": "Belgium",
        "litres": 325
      }, {
        "country": "The Netherlands",
        "litres": 40
      }];

      /* Create axes */
      var categoryAxis = chart.xAxes.push(new am4charts.CategoryAxis());
      categoryAxis.dataFields.category = "country";

      var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());
      valueAxis.renderer.axisFills.template.fill = chart.colors.getIndex(2);
      valueAxis.renderer.axisFills.template.fillOpacity = 0.05;

      /* Create and configure series */
      var series = chart.series.push(new am4charts.RadarSeries());
      series.dataFields.valueY = "litres";
      series.dataFields.categoryX = "country";
      series.name = "Sales";
      series.strokeWidth = 3;

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      }); // end am4core.ready()
   }
   if(jQuery('#am-polar-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      /* Create chart instance */
      var chart = am4core.create("am-polar-chart", am4charts.RadarChart);

      /* Add data */
      chart.data = [ {
        "direction": "N",
        "value": 8
      }, {
        "direction": "NE",
        "value": 9
      }, {
        "direction": "E",
        "value": 4.5
      }, {
        "direction": "SE",
        "value": 3.5
      }, {
        "direction": "S",
        "value": 9.2
      }, {
        "direction": "SW",
        "value": 8.4
      }, {
        "direction": "W",
        "value": 11.1
      }, {
        "direction": "NW",
        "value": 10
      } ];

      /* Create axes */
      var categoryAxis = chart.xAxes.push(new am4charts.CategoryAxis());
      categoryAxis.dataFields.category = "direction";

      var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());
      //valueAxis.renderer.gridType = "polygons";

      var range = categoryAxis.axisRanges.create();
      range.category = "NW";
      range.endCategory = "NW";
      range.axisFill.fill = am4core.color("#4788ff");
      range.axisFill.fillOpacity = 0.3;

      var range2 = categoryAxis.axisRanges.create();
      range2.category = "N";
      range2.endCategory = "N";
      range2.axisFill.fill = am4core.color("#ff4b4b");
      range2.axisFill.fillOpacity = 0.3;

      var range3 = categoryAxis.axisRanges.create();
      range3.category = "SE";
      range3.endCategory = "SW";
      range3.axisFill.fill = am4core.color("#37e6b0");
      range3.axisFill.fillOpacity = 0.3;
      range3.locations.endCategory = 0;

      /* Create and configure series */
      var series = chart.series.push(new am4charts.RadarSeries());
      series.dataFields.valueY = "value";
      series.dataFields.categoryX = "direction";
      series.name = "Wind direction";
      series.strokeWidth = 3;
      series.fillOpacity = 0.2;

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      }); // end am4core.ready()
   }

   if(jQuery('#am-polarscatter-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      /* Create chart instance */
      var chart = am4core.create("am-polarscatter-chart", am4charts.RadarChart);
       chart.colors.list = [am4core.color("#4788ff"),am4core.color("#fe721c"),am4core.color("#37e6b0")];

      /* Add data */
      chart.data = [{
        "country": "Lithuania",
        "litres": 501,
        "units": 250
      }, {
        "country": "Czech Republic",
        "litres": 301,
        "units": 222
      }, {
        "country": "Ireland",
        "litres": 266,
        "units": 179
      }, {
        "country": "Germany",
        "litres": 165,
        "units": 298
      }, {
        "country": "Australia",
        "litres": 139,
        "units": 299
      }];

      /* Create axes */
      var xAxis = chart.xAxes.push(new am4charts.ValueAxis());
      xAxis.renderer.maxLabelPosition = 0.99;

      var yAxis = chart.yAxes.push(new am4charts.ValueAxis());
      yAxis.renderer.labels.template.verticalCenter = "bottom";
      yAxis.renderer.labels.template.horizontalCenter = "right";
      yAxis.renderer.maxLabelPosition = 0.99;
      yAxis.renderer.labels.template.paddingBottom = 1;
      yAxis.renderer.labels.template.paddingRight = 3;

      /* Create and configure series */
      var series1 = chart.series.push(new am4charts.RadarSeries());
      series1.bullets.push(new am4charts.CircleBullet());
      series1.strokeOpacity = 0;
      series1.dataFields.valueX = "x";
      series1.dataFields.valueY = "y";
      series1.name = "Series #1";
      series1.sequencedInterpolation = true;
      series1.sequencedInterpolationDelay = 10;
      series1.data = [
        { "x": 83, "y": 5.1 },
        { "x": 44, "y": 5.8 },
        { "x": 76, "y": 9 },
        { "x": 2, "y": 1.4 },
        { "x": 100, "y": 8.3 },
        { "x": 96, "y": 1.7 },
        { "x": 68, "y": 3.9 },
        { "x": 0, "y": 3 },
        { "x": 100, "y": 4.1 },
        { "x": 16, "y": 5.5 },
        { "x": 71, "y": 6.8 },
        { "x": 100, "y": 7.9 },
        { "x": 35, "y": 8 },
        { "x": 44, "y": 6 },
        { "x": 64, "y": 0.7 },
        { "x": 53, "y": 3.3 },
        { "x": 92, "y": 4.1 },
        { "x": 43, "y": 7.3 },
        { "x": 15, "y": 7.5 },
        { "x": 43, "y": 4.3 },
        { "x": 90, "y": 9.9 }
      ];

      var series2 = chart.series.push(new am4charts.RadarSeries());
      series2.bullets.push(new am4charts.CircleBullet());
      series2.strokeOpacity = 0;
      series2.dataFields.valueX = "x";
      series2.dataFields.valueY = "y";
      series2.name = "Series #2";
      series2.sequencedInterpolation = true;
      series2.sequencedInterpolationDelay = 10;
      series2.data = [
        { "x": 178, "y": 1.3 },
        { "x": 129, "y": 3.4 },
        { "x": 99, "y": 2.4 },
        { "x": 80, "y": 9.9 },
        { "x": 118, "y": 9.4 },
        { "x": 103, "y": 8.7 },
        { "x": 91, "y": 4.2 },
        { "x": 151, "y": 1.2 },
        { "x": 168, "y": 5.2 },
        { "x": 168, "y": 1.6 },
        { "x": 152, "y": 1.2 },
        { "x": 138, "y": 7.7 },
        { "x": 107, "y": 3.9 },
        { "x": 124, "y": 0.7 },
        { "x": 130, "y": 2.6 },
        { "x": 86, "y": 9.2 },
        { "x": 169, "y": 7.5 },
        { "x": 122, "y": 9.9 },
        { "x": 100, "y": 3.8 },
        { "x": 172, "y": 4.1 },
        { "x": 140, "y": 7.3 },
        { "x": 161, "y": 2.3 },
        { "x": 141, "y": 0.9 }
      ];

      var series3 = chart.series.push(new am4charts.RadarSeries());
      series3.bullets.push(new am4charts.CircleBullet());
      series3.strokeOpacity = 0;
      series3.dataFields.valueX = "x";
      series3.dataFields.valueY = "y";
      series3.name = "Series #3";
      series3.sequencedInterpolation = true;
      series3.sequencedInterpolationDelay = 10;
      series3.data = [
        { "x": 419, "y": 4.9 },
        { "x": 417, "y": 5.5 },
        { "x": 434, "y": 0.1 },
        { "x": 344, "y": 2.5 },
        { "x": 279, "y": 7.5 },
        { "x": 307, "y": 8.4 },
        { "x": 279, "y": 9 },
        { "x": 220, "y": 8.4 },
        { "x": 201, "y": 9.7 },
        { "x": 288, "y": 1.2 },
        { "x": 333, "y": 7.4 },
        { "x": 308, "y": 1.9 },
        { "x": 330, "y": 8 },
        { "x": 408, "y": 1.7 },
        { "x": 274, "y": 0.8 },
        { "x": 296, "y": 3.1 },
        { "x": 279, "y": 4.3 },
        { "x": 379, "y": 5.6 },
        { "x": 175, "y": 6.8 }
      ];

      /* Add legend */
      chart.legend = new am4charts.Legend();

      /* Add cursor */
      chart.cursor = new am4charts.RadarCursor();

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      }); // end am4core.ready()
   }
   if(jQuery('#am-3dpie-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      var chart = am4core.create("am-3dpie-chart", am4charts.PieChart3D);
      chart.hiddenState.properties.opacity = 0; // this creates initial fade-in

      chart.legend = new am4charts.Legend();

      chart.data = [
        {
          country: "Lithuania",
          litres: 501.9,
          fill: "red"
        },
        {
          country: "Germany",
          litres: 165.8
        },
        {
          country: "Australia",
          litres: 139.9
        },
        {
          country: "Austria",
          litres: 128.3
        },
        {
          country: "UK",
          litres: 99
        },
        {
          country: "Belgium",
          litres: 60
        }
      ];

      var series = chart.series.push(new am4charts.PieSeries3D());
      series.colors.list = [am4core.color("#4788ff"),am4core.color("#37e6b0"),am4core.color("#ff4b4b"),
      am4core.color("#fe721c"),am4core.color("#876cfe"),am4core.color("#01041b")];
      series.dataFields.value = "litres";
      series.dataFields.category = "country";

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      }); // end am4core.ready()
   }

   if(jQuery('#am-layeredcolumn-chart').length){
      am4core.ready(function() {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      // Create chart instance
      var chart = am4core.create("am-layeredcolumn-chart", am4charts.XYChart);
      chart.colors.list = [am4core.color("#37e6b0"),am4core.color("#4788ff")];

      // Add percent sign to all numbers
      chart.numberFormatter.numberFormat = "#.#'%'";

      // Add data
      chart.data = [{
          "country": "USA",
          "year2004": 3.5,
          "year2005": 4.2
      }, {
          "country": "UK",
          "year2004": 1.7,
          "year2005": 3.1
      }, {
          "country": "Canada",
          "year2004": 2.8,
          "year2005": 2.9
      }, {
          "country": "Japan",
          "year2004": 2.6,
          "year2005": 2.3
      }, {
          "country": "France",
          "year2004": 1.4,
          "year2005": 2.1
      }, {
          "country": "Brazil",
          "year2004": 2.6,
          "year2005": 4.9
      }];

      // Create axes
      var categoryAxis = chart.xAxes.push(new am4charts.CategoryAxis());
      categoryAxis.dataFields.category = "country";
      categoryAxis.renderer.grid.template.location = 0;
      categoryAxis.renderer.minGridDistance = 30;

      var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());
      valueAxis.title.text = "GDP growth rate";
      valueAxis.title.fontWeight = 800;

      // Create series
      var series = chart.series.push(new am4charts.ColumnSeries());
      series.dataFields.valueY = "year2004";
      series.dataFields.categoryX = "country";
      series.clustered = false;
      series.tooltipText = "GDP grow in {categoryX} (2004): [bold]{valueY}[/]";

      var series2 = chart.series.push(new am4charts.ColumnSeries());
      series2.dataFields.valueY = "year2005";
      series2.dataFields.categoryX = "country";
      series2.clustered = false;
      series2.columns.template.width = am4core.percent(50);
      series2.tooltipText = "GDP grow in {categoryX} (2005): [bold]{valueY}[/]";

      chart.cursor = new am4charts.XYCursor();
      chart.cursor.lineX.disabled = true;
      chart.cursor.lineY.disabled = true;

        const body = document.querySelector('body')
        if (body.classList.contains('dark')) {
          amChartUpdate(chart, {
            dark: true
          })
        }

        document.addEventListener('ChangeColorMode', function (e) {
          amChartUpdate(chart, e.detail)
        })

      }); // end am4core.ready()
   }

/*---------------------------------------------------------------------
   Morris Charts
-----------------------------------------------------------------------*/

if(jQuery('#morris-line-chart').length){
  new Morris.Line({
    // ID of the element in which to draw the chart.
    element: 'morris-line-chart',
    // Chart data records -- each entry in this array corresponds to a point on
    // the chart.
    data: [
      { year: '2008', value: 20 },
      { year: '2009', value: 10 },
      { year: '2010', value: 5 },
      { year: '2011', value: 5 },
      { year: '2012', value: 20 }
    ],
    // The name of the data record attribute that contains x-values.
    xkey: 'year',
    // A list of names of data record attributes that contain y-values.
    ykeys: ['value'],
    // Labels for the ykeys -- will be displayed when you hover over the
    // chart.
    labels: ['Value'],
    lineColors: ['#4788ff']
  });
}

if(jQuery('#morris-bar-chart').length){
 Morris.Bar({
element: 'morris-bar-chart',
data: [
  {x: '2011 Q1', y: 3, z: 2, a: 3},
  {x: '2011 Q2', y: 2, z: null, a: 1},
  {x: '2011 Q3', y: 0, z: 2, a: 4},
  {x: '2011 Q4', y: 2, z: 4, a: 3}
],
xkey: 'x',
ykeys: ['y', 'z', 'a'],
labels: ['Y', 'Z', 'A'],
barColors: ['#4788ff', '#fe721c', '#37e6b0'],
hoverCallback: function (index, options, content, row) {
                    return '';
                  }
}).on('click', function(i, row){
console.log(i, row);
});
}

if(jQuery('#morris-area-chart').length){
  var area = new Morris.Area({
    element: 'morris-area-chart',
    resize: true,
    data: [
      {y: '2011 Q1', item1: 2666, item2: 2666},
      {y: '2011 Q2', item1: 2778, item2: 2294},
      {y: '2011 Q3', item1: 4912, item2: 1969},
      {y: '2011 Q4', item1: 3767, item2: 3597},
      {y: '2012 Q1', item1: 6810, item2: 1914},
      {y: '2012 Q2', item1: 5670, item2: 4293},
      {y: '2012 Q3', item1: 4820, item2: 3795},
      {y: '2012 Q4', item1: 15073, item2: 5967},
      {y: '2013 Q1', item1: 10687, item2: 4460},
      {y: '2013 Q2', item1: 8432, item2: 5713}
    ],
    xkey: 'y',
    ykeys: ['item1', 'item2'],
    labels: ['Item 1', 'Item 2'],
    lineColors: ['#75e275', '#75bcff'],
    hoverCallback: function (index, options, content, row) {
                    return '';
                  }
  });
}

if(jQuery('#morris-donut-chart').length){
  var donut = new Morris.Donut({
    element: 'morris-donut-chart',
    resize: true,
    colors: ["#4788ff", "#ff4b4b", "#37e6b0"],
    data: [
      {label: "Download Sales", value: 12},
      {label: "In-Store Sales", value: 30},
      {label: "Mail-Order Sales", value: 20}
    ],
    hideHover: 'auto'
  });
}

/*---------------------------------------------------------------------
   High Charts
-----------------------------------------------------------------------*/
if (jQuery("#high-basicline-chart").length && Highcharts.chart("high-basicline-chart", {
    chart: {
      type: "spline",
      inverted: !0
    },
    title: {
      text: "Atmosphere Temperature by Altitude"
    },
    subtitle: {
      text: "According to the Standard Atmosphere Model"
    },
    xAxis: {
      reversed: !1,
      title: {
        enabled: !0,
        text: "Altitude"
      },
      labels: {
        format: "{value} km"
      },
      maxPadding: .05,
      showLastLabel: !0
    },
    yAxis: {
      title: {
        text: "Temperature"
      },
      labels: {
        format: "{value}°"
      },
      lineWidth: 2
    },
    legend: {
      enabled: !1
    },
    tooltip: {
      headerFormat: "<b>{series.name}</b><br/>",
      pointFormat: "{point.x} km: {point.y}°C"
    },
    plotOptions: {
      spline: {
        marker: {
          enable: !1
        }
      }
    },
    series: [{
      name: "Temperature",
      color: "#4788ff",
      data: [
        [0, 15],
        [10, -50],
        [20, -56.5],
        [30, -46.5],
        [40, -22.1],
        [50, -2.5],
        [60, -27.7],
        [70, -55.7],
        [80, -76.5]
      ]
    }]
  }), jQuery("#high-area-chart").length && Highcharts.chart("high-area-chart", {
    chart: {
      type: "areaspline"
    },
    title: {
      text: "Average fruit consumption during one week"
    },
    legend: {
      layout: "vertical",
      align: "left",
      verticalAlign: "top",
      x: 150,
      y: 100,
      floating: !0,
      borderWidth: 1,
      backgroundColor: Highcharts.defaultOptions.legend.backgroundColor || "#FFFFFF"
    },
    xAxis: {
      categories: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
      plotBands: [{
        from: 4.5,
        to: 6.5,
        color: "rgba(68, 170, 213, .2)"
      }]
    },
    yAxis: {
      title: {
        text: "Fruit units"
      }
    },
    tooltip: {
      shared: !0,
      valueSuffix: " units"
    },
    credits: {
      enabled: !1
    },
    plotOptions: {
      areaspline: {
        fillOpacity: .5
      }
    },
    series: [{
      name: "John",
      color: "#4788ff",
      data: [3, 4, 3, 5, 4, 10, 12]
    }, {
      name: "Jane",
      color: "#37e6b0",
      data: [1, 3, 4, 3, 3, 5, 4]
    }]
  }), jQuery("#high-columnndbar-chart").length && Highcharts.chart("high-columnndbar-chart", {
    chart: {
      type: "bar"
    },
    title: {
      text: "Stacked bar chart"
    },
    xAxis: {
      categories: ["Apples", "Oranges", "Pears", "Grapes", "Bananas"]
    },
    yAxis: {
      min: 0,
      title: {
        text: "Total fruit consumption"
      }
    },
    legend: {
      reversed: !0
    },
    plotOptions: {
      series: {
        stacking: "normal"
      }
    },
    series: [{
      name: "John",
      color: "#4788ff",
      data: [5, 3, 4, 7, 2]
    }, {
      name: "Jane",
      color: "#ff4b4b",
      data: [2, 2, 3, 2, 1]
    }, {
      name: "Joe",
      color: "#37e6b0",
      data: [3, 4, 4, 2, 5]
    }]
  }), jQuery("#high-pie-chart").length && Highcharts.chart("high-pie-chart", {
    chart: {
      plotBackgroundColor: null,
      plotBorderWidth: null,
      plotShadow: !1,
      type: "pie"
    },
    colorAxis: {},
    title: {
      text: "Browser market shares in January, 2018"
    },
    tooltip: {
      pointFormat: "{series.name}: <b>{point.percentage:.1f}%</b>"
    },
    plotOptions: {
      pie: {
        allowPointSelect: !0,
        cursor: "pointer",
        dataLabels: {
          enabled: !0,
          format: "<b>{point.name}</b>: {point.percentage:.1f} %"
        }
      }
    },
    series: [{
      name: "Brands",
      colorByPoint: !0,
      data: [{
        name: "Chrome",
        y: 61.41,
        sliced: !0,
        selected: !0,
        color: "#4788ff"
      }, {
        name: "Internet Explorer",
        y: 11.84,
        color: "#ff4b4b"
      }, {
        name: "Firefox",
        y: 10.85,
        color: "#65f9b3"
      }, {
        name: "Edge",
        y: 4.67,
        color: "#6ce6f4"
      }, {
        name: "Other",
        y: 2.61
      }]
    }]
  }), jQuery("#high-scatterplot-chart").length && Highcharts.chart("high-scatterplot-chart", {
    chart: {
      type: "scatter",
      zoomType: "xy"
    },
    accessibility: {
      description: "A scatter plot compares the height and weight of 507 individuals by gender. Height in centimeters is plotted on the X-axis and weight in kilograms is plotted on the Y-axis. The chart is interactive, and each data point can be hovered over to expose the height and weight data for each individual. The scatter plot is fairly evenly divided by gender with females dominating the left-hand side of the chart and males dominating the right-hand side. The height data for females ranges from 147.2 to 182.9 centimeters with the greatest concentration between 160 and 165 centimeters. The weight data for females ranges from 42 to 105.2 kilograms with the greatest concentration at around 60 kilograms. The height data for males ranges from 157.2 to 198.1 centimeters with the greatest concentration between 175 and 180 centimeters. The weight data for males ranges from 53.9 to 116.4 kilograms with the greatest concentration at around 80 kilograms."
    },
    title: {
      text: "Height Versus Weight of 507 Individuals by Gender"
    },
    subtitle: {
      text: "Source: Heinz  2003"
    },
    xAxis: {
      title: {
        enabled: !0,
        text: "Height (cm)"
      },
      startOnTick: !0,
      endOnTick: !0,
      showLastLabel: !0
    },
    yAxis: {
      title: {
        text: "Weight (kg)"
      }
    },
    legend: {
      layout: "vertical",
      align: "left",
      verticalAlign: "top",
      x: 100,
      y: 70,
      floating: !0,
      backgroundColor: Highcharts.defaultOptions.chart.backgroundColor,
      borderWidth: 1
    },
    plotOptions: {
      scatter: {
        marker: {
          radius: 5,
          states: {
            hover: {
              enabled: !0,
              lineColor: "rgb(100,100,100)"
            }
          }
        },
        states: {
          hover: {
            marker: {
              enabled: !1
            }
          }
        },
        tooltip: {
          headerFormat: "<b>{series.name}</b><br>",
          pointFormat: "{point.x} cm, {point.y} kg"
        }
      }
    },
    series: [{
      name: "Female",
      color: "rgba(223, 83, 83, .5)",
      data: [
        [161.2, 51.6],
        [167.5, 59],
        [159.5, 49.2],
        [157, 63],
        [155.8, 53.6],
        [170, 59],
        [159.1, 47.6],
        [166, 69.8],
        [176.2, 66.8],
        [160.2, 75.2],
        [172.7, 62],
        [155, 49.2],
        [156.5, 67.2],
        [164, 53.8],
        [160.9, 54.4]
      ],
      color: "#4788ff"
    }, {
      name: "Male",
      color: "rgba(119, 152, 191, .5)",
      data: [
        [174, 65.6],
        [175.3, 71.8],
        [193.5, 80.7],
        [186.5, 72.6],
        [187.2, 78.8],
        [181.5, 74.8],
        [184, 86.4],
        [184.5, 78.4],
        [175, 62],
        [184, 81.6],
        [180.1, 93],
        [175.5, 80.9],
        [180.6, 72.7],
        [184.4, 68],
        [175.5, 70.9],
        [180.3, 83.2],
        [180.3, 83.2]
      ],
      color: "#ff4b4b"
    }]
  }), jQuery("#high-linendcolumn-chart").length && Highcharts.chart("high-linendcolumn-chart", {
    chart: {
      zoomType: "xy"
    },
    title: {
      text: "Average Monthly Temperature and Rainfall in Tokyo"
    },
    subtitle: {
      text: "Source: WorldClimate.com"
    },
    xAxis: [{
      categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
      crosshair: !0
    }],
    yAxis: [{
      labels: {
        format: "{value}°C",
        style: {
          color: Highcharts.getOptions().colors[1]
        }
      },
      title: {
        text: "Temperature",
        style: {
          color: Highcharts.getOptions().colors[1]
        }
      }
    }, {
      title: {
        text: "Rainfall",
        style: {
          color: Highcharts.getOptions().colors[0]
        }
      },
      labels: {
        format: "{value} mm",
        style: {
          color: Highcharts.getOptions().colors[0]
        }
      },
      opposite: !0
    }],
    tooltip: {
      shared: !0
    },
    legend: {
      layout: "vertical",
      align: "left",
      x: 120,
      verticalAlign: "top",
      y: 100,
      floating: !0,
      backgroundColor: Highcharts.defaultOptions.legend.backgroundColor || "rgba(255,255,255,0.25)"
    },
    series: [{
      name: "Rainfall",
      type: "column",
      yAxis: 1,
      data: [49.9, 71.5, 106.4, 129.2, 144, 176, 135.6, 148.5, 216.4, 194.1, 95.6, 54.4],
      color: "#4788ff",
      tooltip: {
        valueSuffix: " mm"
      }
    }, {
      name: "Temperature",
      type: "spline",
      data: [7, 6.9, 9.5, 14.5, 18.2, 21.5, 25.2, 26.5, 23.3, 18.3, 13.9, 9.6],
      color: "#ff4b4b",
      tooltip: {
        valueSuffix: "°C"
      }
    }]
  }), jQuery("#high-dynamic-chart").length && Highcharts.chart("high-dynamic-chart", {
    chart: {
      type: "spline",
      animation: Highcharts.svg,
      marginRight: 10,
      events: {
        load: function() {
          var e = this.series[0];
          setInterval(function() {
            var t = (new Date).getTime(),
              a = Math.random();
            e.addPoint([t, a], !0, !0)
          }, 1e3)
        }
      }
    },
    time: {
      useUTC: !1
    },
    title: {
      text: "Live random data"
    },
    accessibility: {
      announceNewData: {
        enabled: !0,
        minAnnounceInterval: 15e3,
        announcementFormatter: function(e, t, a) {
          return !!a && "New point added. Value: " + a.y
        }
      }
    },
    xAxis: {
      type: "datetime",
      tickPixelInterval: 150
    },
    yAxis: {
      title: {
        text: "Value"
      },
      plotLines: [{
        value: 0,
        width: 1,
        color: "#808080"
      }]
    },
    tooltip: {
      headerFormat: "<b>{series.name}</b><br/>",
      pointFormat: "{point.x:%Y-%m-%d %H:%M:%S}<br/>{point.y:.2f}"
    },
    legend: {
      enabled: !1
    },
    exporting: {
      enabled: !1
    },
    series: [{
      name: "Random data",
      color: "#4788ff",
      data: function() {
        var e, t = [],
          a = (new Date).getTime();
        for (e = -19; e <= 0; e += 1) t.push({
          x: a + 1e3 * e,
          y: Math.random()
        });
        return t
      }()
    }]
  }), jQuery("#high-3d-chart").length) {
  var chart = new Highcharts.Chart({
    chart: {
      renderTo: "high-3d-chart",
      type: "column",
      options3d: {
        enabled: !0,
        alpha: 15,
        beta: 15,
        depth: 50,
        viewDistance: 25
      }
    },
    title: {
      text: "Chart rotation demo"
    },
    subtitle: {
      text: "Test options by dragging the sliders below"
    },
    plotOptions: {
      column: {
        depth: 25
      }
    },
    series: [{
      data: [29.9, 71.5, 106.4, 129.2, 144, 176, 135.6, 148.5, 216.4, 194.1, 95.6, 54.4],
      color: "#4788ff"
    }]
  });

  function showValues() {
    $("#alpha-value").html(chart.options.chart.options3d.alpha), $("#beta-value").html(chart.options.chart.options3d.beta), $("#depth-value").html(chart.options.chart.options3d.depth)
  }
  $("#sliders input").on("input change", function() {
    chart.options.chart.options3d[this.id] = parseFloat(this.value), showValues(), chart.redraw(!1)
  }), showValues()
}
if (jQuery("#high-gauges-chart").length && Highcharts.chart("high-gauges-chart", {
    chart: {
      type: "gauge",
      plotBackgroundColor: null,
      plotBackgroundImage: null,
      plotBorderWidth: 0,
      plotShadow: !1
    },
    title: {
      text: "Speedometer"
    },
    pane: {
      startAngle: -150,
      endAngle: 150,
      background: [{
        backgroundColor: {
          linearGradient: {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 1
          },
          stops: [
            [0, "#FFF"],
            [1, "#333"]
          ]
        },
        borderWidth: 0,
        outerRadius: "109%"
      }, {
        backgroundColor: {
          linearGradient: {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 1
          },
          stops: [
            [0, "#333"],
            [1, "#FFF"]
          ]
        },
        borderWidth: 1,
        outerRadius: "107%"
      }, {}, {
        backgroundColor: "#DDD",
        borderWidth: 0,
        outerRadius: "105%",
        innerRadius: "103%"
      }]
    },
    yAxis: {
      min: 0,
      max: 200,
      minorTickInterval: "auto",
      minorTickWidth: 1,
      minorTickLength: 10,
      minorTickPosition: "inside",
      minorTickColor: "#666",
      tickPixelInterval: 30,
      tickWidth: 2,
      tickPosition: "inside",
      tickLength: 10,
      tickColor: "#666",
      labels: {
        step: 2,
        rotation: "auto"
      },
      title: {
        text: "km/h"
      },
      plotBands: [{
        from: 0,
        to: 120,
        color: "#55BF3B"
      }, {
        from: 120,
        to: 160,
        color: "#DDDF0D"
      }, {
        from: 160,
        to: 200,
        color: "#DF5353"
      }]
    },
    series: [{
      name: "Speed",
      data: [80],
      tooltip: {
        valueSuffix: " km/h"
      }
    }]
  }, function(e) {
    e.renderer.forExport || setInterval(function() {
      var t, a = e.series[0].points[0],
        n = Math.round(20 * (Math.random() - .5));
      ((t = a.y + n) < 0 || t > 200) && (t = a.y - n), a.update(t)
    }, 3e3)
  }), jQuery("#high-barwithnagative-chart").length) {
  var categories = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95-99", "100 + "];
  Highcharts.chart("high-barwithnagative-chart", {
    chart: {
      type: "bar"
    },
    title: {
      text: "Population pyramid for Germany, 2018"
    },
    subtitle: {
      text: 'Source: <a href="http://populationpyramid.net/germany/2018/">Population Pyramids of the World from 1950 to 2100</a>'
    },
    accessibility: {
      point: {
        descriptionFormatter: function(e) {
          return e.index + 1 + ", Age " + e.category + ", " + Math.abs(e.y) + "%. " + e.series.name + "."
        }
      }
    },
    xAxis: [{
      categories: categories,
      reversed: !1,
      labels: {
        step: 1
      },
      accessibility: {
        description: "Age (male)"
      }
    }, {
      opposite: !0,
      reversed: !1,
      categories: categories,
      linkedTo: 0,
      labels: {
        step: 1
      },
      accessibility: {
        description: "Age (female)"
      }
    }],
    yAxis: {
      title: {
        text: null
      },
      labels: {
        formatter: function() {
          return Math.abs(this.value) + "%"
        }
      },
      accessibility: {
        description: "Percentage population",
        rangeDescription: "Range: 0 to 5%"
      }
    },
    plotOptions: {
      series: {
        stacking: "normal"
      }
    },
    tooltip: {
      formatter: function() {
        return "<b>" + this.series.name + ", age " + this.point.category + "</b><br/>Population: " + Highcharts.numberFormat(Math.abs(this.point.y), 1) + "%"
      }
    },
    series: [{
      name: "Male",
      data: [-2.2, -2.1, -2.2, -2.4, -2.7, -3, -3.3, -3.2, -2.9, -3.5, -4.4, -4.1, -0],
      color: "#4788ff"
    }, {
      name: "Female",
      data: [2.1, 2, 2.1, 2.3, 2.6, 2.9, 3.2, 3.1, 2.9, 3.4, 0],
      color: "#ff4b4b"
    }]
  })
}

/*--------------Widget Chart 1----------------*/

var options = {
    chart: {
        height: 80,
        type: 'area',
        sparkline: {
            enabled: true
        },
        group: 'sparklines',

    },
    dataLabels: {
        enabled: false
    },
    stroke: {
        width: 3,
        curve: 'smooth'
    },
    fill: {
        type: 'gradient',
        gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.5,
            opacityTo: 0,
        }
    },

    series: [{
        name: 'series1',
        data: [60, 15, 50, 30, 70]
    }, ],
    colors: ['#50b5ff'],

    xaxis: {
        type: 'datetime',
        categories: ["2018-08-19T00:00:00", "2018-09-19T01:30:00", "2018-10-19T02:30:00", "2018-11-19T01:30:00", "2018-12-19T01:30:00"],
    },
    tooltip: {
        x: {
            format: 'dd/MM/yy HH:mm'
        },
    }
};

if(jQuery('#chart-1').length){
    var chart = new ApexCharts(
        document.querySelector("#chart-1"),
        options
    );
    chart.render();
}


/*--------------Widget Chart 2----------------*/
var options = {
    chart: {
        height: 80,
        type: 'area',
        sparkline: {
            enabled: true
        },
        group: 'sparklines',

    },
    dataLabels: {
        enabled: false
    },
    stroke: {
        width: 3,
        curve: 'smooth'
    },
    fill: {
        type: 'gradient',
        gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.5,
            opacityTo: 0,
        }
    },
    series: [{
        name: 'series1',
        data: [70, 40, 60, 30, 60]
    }, ],
    colors: ['#fd7e14'],

    xaxis: {
        type: 'datetime',
        categories: ["2018-08-19T00:00:00", "2018-09-19T01:30:00", "2018-10-19T02:30:00", "2018-11-19T01:30:00", "2018-12-19T01:30:00"],
    },
    tooltip: {
        x: {
            format: 'dd/MM/yy HH:mm'
        },
    }
};

if(jQuery('#chart-2').length){
    var chart = new ApexCharts(
        document.querySelector("#chart-2"),
        options
    );

    chart.render();
}

/*--------------Widget Chart 3----------------*/
var options = {
    chart: {
        height: 80,
        type: 'area',
        sparkline: {
            enabled: true
        },
        group: 'sparklines',

    },
    dataLabels: {
        enabled: false
    },
    stroke: {
        width: 3,
        curve: 'smooth'
    },
    fill: {
        type: 'gradient',
        gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.5,
            opacityTo: 0,
        }
    },
    series: [{
        name: 'series1',
        data: [60, 40, 60, 40, 70]
    }, ],
    colors: ['#49f0d3'],

    xaxis: {
        type: 'datetime',
        categories: ["2018-08-19T00:00:00", "2018-09-19T01:30:00", "2018-10-19T02:30:00", "2018-11-19T01:30:00", "2018-12-19T01:30:00"],
    },
    tooltip: {
        x: {
            format: 'dd/MM/yy HH:mm'
        },
    }
};
if(jQuery('#chart-3').length){
    var chart = new ApexCharts(
        document.querySelector("#chart-3"),
        options
    );
    chart.render();
}

/*--------------Widget Chart 4----------------*/
var options = {
    chart: {
        height: 80,
        type: 'area',
        sparkline: {
            enabled: true
        },
        group: 'sparklines',

    },
    dataLabels: {
        enabled: false
    },
    stroke: {
        width: 3,
        curve: 'smooth'
    },
    fill: {
        type: 'gradient',
        gradient: {
            shadeIntensity: 1,
            opacityFrom: 0.5,
            opacityTo: 0,
        }
    },
    series: [{
        name: 'series1',
        data: [75, 30, 60, 35, 60]
    }, ],
    colors: ['#ff9b8a'],

    xaxis: {
        type: 'datetime',
        categories: ["2018-08-19T00:00:00", "2018-09-19T01:30:00", "2018-10-19T02:30:00", "2018-11-19T01:30:00", "2018-12-19T01:30:00"],
    },
    tooltip: {
        x: {
            format: 'dd/MM/yy HH:mm'
        },
    }
};

if(jQuery('#chart-4').length){
    var chart = new ApexCharts(
        document.querySelector("#chart-4"),
        options
    );

    chart.render();
}

/*--------------Widget Box----------------*/

if(jQuery('#iq-chart-box1').length){
    var options = {
      series: [{
        name: "Total sales",
        data: [10, 10, 35, 10]
    }],
      colors: ["#50b5ff"],
      chart: {
      height: 50,
      width: 100,
      type: 'line',
      sparkline: {
          enabled: true,
      },
      zoom: {
        enabled: false
      }
    },
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'straight'
    },
    title: {
      text: '',
      align: 'left'
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
        opacity: 0.5
      },
    },
    xaxis: {
      categories: ['Jan', 'Feb', 'Mar'],
    }
    };

    var chart = new ApexCharts(document.querySelector("#iq-chart-box1"), options);
    chart.render();
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if(jQuery('#iq-chart-box2').length){
    var options = {
      series: [{
        name: "Sale Today",
        data: [10, 10, 35, 10]
    }],
      colors: ["#ff9b8a"],
      chart: {
      height: 50,
      width: 100,
      type: 'line',
      sparkline: {
          enabled: true,
      },
      zoom: {
        enabled: false
      }
    },
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'straight'
    },
    title: {
      text: '',
      align: 'left'
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
        opacity: 0.5
      },
    },
    xaxis: {
      categories: ['Jan', 'Feb', 'Mar'],
    }
    };

    var chart = new ApexCharts(document.querySelector("#iq-chart-box2"), options);
    chart.render();
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if(jQuery('#iq-chart-box3').length){
    var options = {
      series: [{
        name: "Total Classon",
        data: [10, 10, 35, 10]
    }],
      colors: ["#49f0d3"],
      chart: {
      height: 50,
      width: 100,
      type: 'line',
      sparkline: {
          enabled: true,
      },
      zoom: {
        enabled: false
      }
    },
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'straight'
    },
    title: {
      text: '',
      align: 'left'
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
        opacity: 0.5
      },
    },
    xaxis: {
      categories: ['Jan', 'Feb', 'Mar'],
    }
    };

    var chart = new ApexCharts(document.querySelector("#iq-chart-box3"), options);
    chart.render();
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if(jQuery('#iq-chart-box4').length){
    var options = {
      series: [{
        name: "Total Profit",
        data: [10, 10, 35, 10]
    }],
      colors: ["#fd7e14"],
      chart: {
      height: 50,
      width: 100,
      type: 'line',
      sparkline: {
          enabled: true,
      },
      zoom: {
        enabled: false
      }
    },
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'straight'
    },
    title: {
      text: '',
      align: 'left'
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
        opacity: 0.5
      },
    },
    xaxis: {
      categories: ['Jan', 'Feb', 'Mar'],
    }
    };

    var chart = new ApexCharts(document.querySelector("#iq-chart-box4"), options);
    chart.render();
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
/*--------------Widget End----------------*/

/*-------------- Widget Chart ----------------*/
if (jQuery("#ethernet-chart-01").length) {
    var options = {
      series: [{
        name: "Desktops",
        data: [5, 30, 6, 20, 5, 18, 10]
    }],
    colors: ['#4788ff'],
      chart: {
      height: 60,
      width: 100,
      type: 'line',
      zoom: {
        enabled: false
      },
      sparkline: {
        enabled: true,
      }
    },
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'smooth',
      width: 3
    },
    title: {
      text: '',
      align: 'left'
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
        opacity: 0.5
      },
    },
    xaxis: {
      categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    }
    };

    var chart = new ApexCharts(document.querySelector("#ethernet-chart-01"), options);
    chart.render();
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#ethernet-chart-02").length) {
    var options = {
      series: [{
        name: "Desktops",
        data: [5, 20, 4, 18, 3, 15, 10]
    }],
    colors: ['#1ee2ac'],
      chart: {
      height: 60,
      width: 100,
      type: 'line',
      zoom: {
        enabled: false
      },
      sparkline: {
        enabled: true,
      }
    },
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'smooth',
      width: 3
    },
    title: {
      text: '',
      align: 'left'
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
        opacity: 0.5
      },
    },
    xaxis: {
      categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    }
    };

    var chart = new ApexCharts(document.querySelector("#ethernet-chart-02"), options);
    chart.render();
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#ethernet-chart-03").length) {
    var options = {
      series: [{
        name: "Desktops",
        data: [5, 20, 6, 18, 5, 15, 4]
    }],
    colors: ['#ff4b4b'],
      chart: {
      height: 60,
      width: 100,
      type: 'line',
      zoom: {
        enabled: false
      },
      sparkline: {
        enabled: true,
      }
    },
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'smooth',
      width: 3
    },
    title: {
      text: '',
      align: 'left'
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
        opacity: 0.5
      },
    },
    xaxis: {
      categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    }
    };

    var chart = new ApexCharts(document.querySelector("#ethernet-chart-03"), options);
    chart.render();
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}
if (jQuery("#ethernet-chart-04").length) {
    var options = {
      series: [{
        name: "Desktops",
        data: [5, 15, 3, 20, 5, 18, 13]
    }],
    colors: ['#4788ff'],
      chart: {
      height: 60,
      width: 100,
      type: 'line',
      zoom: {
        enabled: false
      },
      sparkline: {
        enabled: true,
      }
    },
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'smooth',
      width: 3
    },
    title: {
      text: '',
      align: 'left'
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
        opacity: 0.5
      },
    },
    xaxis: {
      categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    }
    };

    var chart = new ApexCharts(document.querySelector("#ethernet-chart-04"), options);
    chart.render();
  const body = document.querySelector('body')
  if (body.classList.contains('dark')) {
    apexChartUpdate(chart, {
      dark: true
    })
  }

  document.addEventListener('ChangeColorMode', function (e) {
    apexChartUpdate(chart, e.detail)
  })
}

/*-------------- Widget Chart End ----------------*/

/*--------------Widget Chart ----------------*/
function getNewSeries(e, t) {
  var a = e + TICKINTERVAL;
  lastDate = a;
  for (var n = 0; n < data.length - 10; n++) data[n].x = a - XAXISRANGE - TICKINTERVAL, data[n].y = 0;
  data.push({
    x: a,
    y: Math.floor(Math.random() * (t.max - t.min + 1)) + t.min
  })
}
if (jQuery("#chart-9").length) {
    var chart9 = new ApexCharts(document.querySelector("#chart-9"), options);
    chart9.render(), window.setInterval(function() {
        getNewSeries(lastDate, {
            min: 10,
            max: 90
        }), jQuery("#chart-9").length && chart9.updateSeries([{
            data: data
        }])
    }, 1e3)
}

function generateData(e, t, a) {
    for (var n = 0, o = []; n < t;) {
        var r = Math.floor(750 * Math.random()) + 1,
            i = Math.floor(Math.random() * (a.max - a.min + 1)) + a.min,
            c = Math.floor(61 * Math.random()) + 15;
        o.push([r, i, c]), 864e5, n++
    }
    return o
}
options = {
    chart: {
        height: 440,
        type: "bubble",
        stacked: !1,
        toolbar: {
            show: !1
        },
        animations: {
            enabled: !0,
            easing: "linear",
            dynamicAnimation: {
                speed: 1e3
            }
        },
        sparkline: {
            enabled: !0
        },
        group: "sparklines"
    },
    dataLabels: {
        enabled: !1
    },
    series: [{
        name: "Bubble1",
        data: generateData(new Date("11 Feb 2017 GMT").getTime(), 10, {
            min: 10,
            max: 60
        })
    }, {
        name: "Bubble2",
        data: generateData(new Date("11 Feb 2017 GMT").getTime(), 10, {
            min: 10,
            max: 60
        })
    }, {
        name: "Bubble3",
        data: generateData(new Date("11 Feb 2017 GMT").getTime(), 10, {
            min: 10,
            max: 60
        })
    }, {
        name: "Bubble4",
        data: generateData(new Date("11 Feb 2017 GMT").getTime(), 10, {
            min: 10,
            max: 60
        })
    }],
    fill: {
        opacity: .8
    },
    title: {
        show: !1
    },
    xaxis: {
        tickAmount: 8,
        type: "category",
        labels: {
            show: !1
        }
    },
    yaxis: {
        max: 70,
        labels: {
            show: !1
        }
    },
    legend: {
        show: !1
    }
};

/*-------------- Widget Chart End ----------------*/
/*--------index-----*/
if (jQuery("#site-trafic-chart").length) {
  var options = {
          series: [{
            name: "series1",
            data: [0, 70, 30, 90, 80, 150]
        }, {
          name: 'series2',
          data: [0, 20, 90, 70, 130, 110]
        }],
        colors: ['#fe721c','#4788ff'],
          chart: {
          height: 365,
          type: 'line',
          zoom: {
            enabled: false
          }
        },
        dataLabels: {
          enabled: false
        },
        stroke: {
          curve: 'smooth'
        },
        title: {
          text: 'Product Trends by Month',
          align: 'left'
        },
        grid: {
          row: {
            colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
            opacity: 0
          },
        },
        xaxis: {
          categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
        },
        yaxis: {
          title: {
            text: ''
          },
          labels: {
            offsetX: -20,
            offsetY: 0
          },
        }
        };

        if(typeof ApexCharts !== typeof undefined){
          var chart = new ApexCharts(document.querySelector("#site-trafic-chart"), options);
          chart.render();
          const body = document.querySelector('body')
          if (body.classList.contains('dark')) {
            apexChartUpdate(chart, {
              dark: true
            })
          }

          document.addEventListener('ChangeColorMode', function (e) {
            apexChartUpdate(chart, e.detail)
          })
        }
}

/*---------------------------------------------------------------------
   Editable Table
-----------------------------------------------------------------------*/
const $tableID = $('#table');
 const $BTN = $('#export-btn');
 const $EXPORT = $('#export');

 const newTr = `
<tr class="hide">
  <td class="pt-3-half" contenteditable="true">Example</td>
  <td class="pt-3-half" contenteditable="true">Example</td>
  <td class="pt-3-half" contenteditable="true">Example</td>
  <td class="pt-3-half" contenteditable="true">Example</td>
  <td class="pt-3-half" contenteditable="true">Example</td>
  <td class="pt-3-half">
    <span class="table-up"><a href="#!" class="indigo-text"><i class="fas fa-long-arrow-alt-up" aria-hidden="true"></i></a></span>
    <span class="table-down"><a href="#!" class="indigo-text"><i class="fas fa-long-arrow-alt-down" aria-hidden="true"></i></a></span>
  </td>
  <td>
    <span class="table-remove"><button type="button" class="btn btn-danger btn-rounded btn-sm my-0 waves-effect waves-light">Remove</button></span>
  </td>
</tr>`;

 $('.table-add').on('click', 'i', () => {

   const $clone = $tableID.find('tbody tr').last().clone(true).removeClass('hide table-line');

   if ($tableID.find('tbody tr').length === 0) {

     $('tbody').append(newTr);
   }

   $tableID.find('table').append($clone);
 });

 $tableID.on('click', '.table-remove', function () {

   $(this).parents('tr').detach();
 });

 $tableID.on('click', '.table-up', function () {

   const $row = $(this).parents('tr');

   if ($row.index() === 1) {
     return;
   }

   $row.prev().before($row.get(0));
 });

 $tableID.on('click', '.table-down', function () {

   const $row = $(this).parents('tr');
   $row.next().after($row.get(0));
 });

 // A few jQuery helpers for exporting only
 jQuery.fn.pop = [].pop;
 jQuery.fn.shift = [].shift;

 $BTN.on('click', () => {

   const $rows = $tableID.find('tr:not(:hidden)');
   const headers = [];
   const data = [];

   // Get the headers (add special header logic here)
   $($rows.shift()).find('th:not(:empty)').each(function () {

     headers.push($(this).text().toLowerCase());
   });

   // Turn all existing rows into a loopable array
   $rows.each(function () {
     const $td = $(this).find('td');
     const h = {};

     // Use the headers from earlier to name our hash keys
     headers.forEach((header, i) => {

       h[header] = $td.eq(i).text();
     });

     data.push(h);
   });

   // Output the result
   $EXPORT.text(JSON.stringify(data));
 });

/*---------------------------------------------------------------------
    Form Wizard - 1
-----------------------------------------------------------------------*/

$(document).ready(function(){

    var current_fs, next_fs, previous_fs; //fieldsets
    var opacity;
    var current = 1;
    var steps = $("fieldset").length;

    setProgressBar(current);

    $(".next").click(function(){

    current_fs = $(this).parent();
    next_fs = $(this).parent().next();

    //Add Class Active
    $("#top-tab-list li").eq($("fieldset").index(next_fs)).addClass("active");
    $("#top-tab-list li").eq($("fieldset").index(current_fs)).addClass("done");

    //show the next fieldset
    next_fs.show();
    //hide the current fieldset with style
    current_fs.animate({opacity: 0}, {
    step: function(now) {
    // for making fielset appear animation
    opacity = 1 - now;

    current_fs.css({
    'display': 'none',
    'position': 'relative',

    });

    next_fs.css({'opacity': opacity});
    },
    duration: 500
    });
    setProgressBar(++current);
    });

    $(".previous").click(function(){

    current_fs = $(this).parent();
    previous_fs = $(this).parent().prev();

    //Remove class active
    $("#top-tab-list li").eq($("fieldset").index(current_fs)).removeClass("active");
    $("#top-tab-list li").eq($("fieldset").index(previous_fs)).removeClass("done");

    //show the previous fieldset
    previous_fs.show();

    //hide the current fieldset with style
    current_fs.animate({opacity: 0}, {
    step: function(now) {
    // for making fielset appear animation
    opacity = 1 - now;

    current_fs.css({
    'display': 'none',
    'position': 'relative'
    });
    previous_fs.css({'opacity': opacity});
    },
    duration: 500
    });
    setProgressBar(--current);
    });

    function setProgressBar(curStep){
    var percent = parseFloat(100 / steps) * curStep;
    percent = percent.toFixed();
    $(".progress-bar")
    .css("width",percent+"%")
    }

    $(".submit").click(function(){
    return false;
    })

});

 /*---------------------------------------------------------------------
   validate form wizard
-----------------------------------------------------------------------*/

$(document).ready(function () {

    var navListItems = $('div.setup-panel div a'),
            allWells = $('.setup-content'),
            allNextBtn = $('.nextBtn');

    allWells.hide();

    navListItems.click(function (e) {
        e.preventDefault();
        var $target = $($(this).attr('href')),
                $item = $(this);

        if (!$item.hasClass('disabled')) {
            navListItems.addClass('active');
            $item.parent().addClass('active');
            allWells.hide();
            $target.show();
            $target.find('input:eq(0)').focus();
        }
    });

    allNextBtn.click(function(){
        var curStep = $(this).closest(".setup-content"),
            curStepBtn = curStep.attr("id"),
            nextStepWizard = $('div.setup-panel div a[href="#' + curStepBtn + '"]').parent().next().children("a"),
            curInputs = curStep.find("input[type='text'],input[type='email'],input[type='password'],input[type='url'],textarea"),
            isValid = true;

        $(".form-group").removeClass("has-error");
        for(var i=0; i<curInputs.length; i++){
            if (!curInputs[i].validity.valid){
                isValid = false;
                $(curInputs[i]).closest(".form-group").addClass("has-error");
            }
        }

        if (isValid)
            nextStepWizard.removeAttr('disabled').trigger('click');
    });

    $('div.setup-panel div a.active').trigger('click');
});
 /*---------------------------------------------------------------------
   Vertical form wizard
-----------------------------------------------------------------------*/
$(document).ready(function(){

    var current_fs, next_fs, previous_fs; //fieldsets
    var opacity;
    var current = 1;
    var steps = $("fieldset").length;

    setProgressBar(current);

    $(".next").click(function(){

    current_fs = $(this).parent();
    next_fs = $(this).parent().next();

    //Add Class Active
    $("#top-tabbar-vertical li").eq($("fieldset").index(next_fs)).addClass("active");

    //show the next fieldset
    next_fs.show();
    //hide the current fieldset with style
    current_fs.animate({opacity: 0}, {
    step: function(now) {
    // for making fielset appear animation
    opacity = 1 - now;

    current_fs.css({
    'display': 'none',
    'position': 'relative'
    });
    next_fs.css({'opacity': opacity});
    },
    duration: 500
    });
    setProgressBar(++current);
    });

    $(".previous").click(function(){

    current_fs = $(this).parent();
    previous_fs = $(this).parent().prev();

    //Remove class active
    $("#top-tabbar-vertical li").eq($("fieldset").index(current_fs)).removeClass("active");

    //show the previous fieldset
    previous_fs.show();

    //hide the current fieldset with style
    current_fs.animate({opacity: 0}, {
    step: function(now) {
    // for making fielset appear animation
    opacity = 1 - now;

    current_fs.css({
    'display': 'none',
    'position': 'relative'
    });
    previous_fs.css({'opacity': opacity});
    },
    duration: 500
    });
    setProgressBar(--current);
    });

    function setProgressBar(curStep){
    var percent = parseFloat(100 / steps) * curStep;
    percent = percent.toFixed();
    $(".progress-bar")
    .css("width",percent+"%")
    }

    $(".submit").click(function(){
    return false;
    })

});


/*---------------------------------------------------------------------
   Profile Image Edit
-----------------------------------------------------------------------*/

$(document).ready(function() {
    var readURL = function(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                $('.profile-pic').attr('src', e.target.result);
            }

            reader.readAsDataURL(input.files[0]);
        }
    }


    $(".file-upload").on('change', function(){
        readURL(this);
    });

    $(".upload-button").on('click', function() {
       $(".file-upload").click();
    });
});

// ratting
$(function() {
  if(typeof $.fn.barrating !== typeof undefined){
    $('#example').barrating({
      theme: 'fontawesome-stars'
    });
    $('#bars-number').barrating({
      theme: 'bars-1to10'
    });
    $('#movie-rating').barrating('show',{
      theme: 'bars-movie'
    });
    $('#movie-rating').barrating('set', 'Mediocre');
    $('#pill-rating').barrating({
      theme: 'bars-pill',
      showValues: true,
      showSelectedRating: false,
      onSelect: function(value, text) {
        alert('Selected rating: ' + value);
    }
    });
  } 
  if (typeof $.fn.mdbRate !== typeof undefined) {
    $('#rateMe1').mdbRate();
    $('#face-rating').mdbRate();
  }
});

// quill
if (jQuery("#editor").length) {
  var quill = new Quill('#editor', {
  theme: 'snow'
  });
}
  // With Tooltip
  if (jQuery("#quill-toolbar").length) {
  var quill = new Quill('#quill-toolbar', {
      modules: {
        toolbar: '#quill-tool'
      },
      placeholder: 'Compose an epic...',
      theme: 'snow'
  });
  // Enable all tooltips
  $('[data-toggle="tooltip"]').tooltip();

  // Can control programmatically too
  $('.ql-italic').mouseover();
  setTimeout(function() {
      $('.ql-italic').mouseout();
  }, 2500);
}
  // file upload
  $(".custom-file-input").on("change", function() {
    var fileName = $(this).val().split("\\").pop();
    $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
  });
  
  //  gallary grid
if(typeof $.fn.magnificPopup !== typeof undefined){
  $('.image-popup-vertical-fit').magnificPopup({
      type: 'image',
      mainClass: 'mfp-with-zoom',
      gallery: {
        enabled: true
      },
      zoom: {
        enabled: true,
        duration: 300,
        easing: 'ease-in-out',
        opener: function (openerElement) {
            return openerElement.is('img') ? openerElement : openerElement.find('img');
        }
      }
  });
}

  // gallary masanarry

    // var $grid = $('.masonry').masonry({
    //      itemSelector: '.item',
    //     columnWidth: '.grid-sizer',
    //     percentPosition: true
    // });
    // $grid.imagesLoaded().progress(function () {
    //     $grid.masonry('layout');
    // });
    if(typeof $.fn.magnificPopup !== typeof undefined){
      $('.gallery').magnificPopup({
            delegate: 'a',
            type: 'image',
            gallery: {
              enabled: true,
              navigateByImgClick: true,
              preload: [0, 1]
            }
      });
    }

//layou-1

  if (jQuery("#layout-1-chart-01").length) {
  var options = {
    series: [70, 30],
    chart: {
    height: 300,
    type: 'radialBar',
  },
  plotOptions: {
    radialBar: {
      dataLabels: {
        name: {
          fontSize: '22px',
        },
        value: {
          fontSize: '16px',
        },
        total: {
          show: true,
          label: 'Total',
          formatter: function (w) {
            return 249
          }
        }
      },
      track: {
        	strokeWidth: '42%',
        },
        
    }
  },
    colors: ['#05bbc9', '#876cfe'],
    stroke: {
      lineCap: "round",
    },
  };

    var chart = new ApexCharts(document.querySelector("#layout-1-chart-01"), options);
  chart.render();
   const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })
}
  if (jQuery("#layout-1-chart-02").length) {
    var options = {
      series: [{
      data: [18, 22, 28, 45,34,20]
    }],
      chart: {
      height: 200,
      type: 'bar',
      events: {
        click: function(chart, w, e) {
        }
      },
      sparkline: {
        enabled: true,
      }
    },
      colors: ["#fe721c", "#4788ff", "#05bbc9", "#876cfe", "#00cc96", "#e72c30"],
    plotOptions: {
      bar: {
        columnWidth: '35%',
        distributed: true
      }
    },
    dataLabels: {
      enabled: false
    },
    legend: {
      show: false
    },
    xaxis: {
      categories: [
        ['India'],
        ['U.S.A'],
        ['Canada'],
        ['Africa'], 
      ],
      labels: {
        style: {
          colors: ["#4788ff","#37e6b0","#fe721c","#876cfe"],
          fontSize: '12px'
        }
      }
    }
  };

    var chart = new ApexCharts(document.querySelector("#layout-1-chart-02"), options);
  chart.render();
    const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })
}
  if (jQuery("#layout-1-chart-03").length) {
    var options = {
      series: [{
        name: ' Clicks',
        type: 'column',
        data: [23, 11, 22, 27, 13, 22, 37, 21]
      }, {
          name: ' Sales',
        type: 'area',
        data: [44, 55, 41, 67, 22, 43, 21, 41]
      }, {
          name: 'Commission',
        type: 'line',
        data: [30, 25, 36, 30, 45, 35, 64, 52]
      }],
      chart: {
        height: 370,
        type: 'line',
        stacked: false,
      },
      stroke: {
        width: [0, 2, 5],
        curve: 'smooth'
      },
      plotOptions: {
        bar: {
          columnWidth: '50%'
        }
      },

      fill: {
        opacity: [0.85, 0.25, 1],
        gradient: {
          inverseColors: false,
          shade: 'light',
          type: "vertical",
          opacityFrom: 0.85,
          opacityTo: 0.55,
          stops: [0, 100, 100, 100]
        }
      },
      labels: ['01/01/2019', '02/01/2019', '03/01/2019', '04/01/2019', '05/01/2019', '06/01/2019', '07/01/2019',
        '08/01/2019'
      ],
      colors: ['#05bbc9', '#fe721c', '#00cc96'],
      markers: {
        size: 0
      },
      xaxis: {
        type: 'datetime',
      },
      yaxis: {
        show: true,
        labels: {
          minWidth: 20,
          maxWidth: 20
        }
      },
      tooltip: {
        shared: true,
        intersect: false,
        y: {
          formatter: function (y) {
            if (typeof y !== "undefined") {
              return y.toFixed(0) + " points";
            }
            return y;

          }
        }
      }
    };

    var chart = new ApexCharts(document.querySelector("#layout-1-chart-03"), options);
    chart.render();
    const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })
  }


  if (jQuery('#layout-1-chart-04').length) {
    am4core.ready(function () {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      var continents = {
        "AF": 0,
        "AN": 1,
        "AS": 2,
        "EU": 3,
        "NA": 4,
        "OC": 5,
        "SA": 6
      }

      // Create map instance
      var chart = am4core.create("layout-1-chart-04", am4maps.MapChart);
      chart.projection = new am4maps.projections.Miller();

      // Create map polygon series for world map
      var worldSeries = chart.series.push(new am4maps.MapPolygonSeries());
      worldSeries.useGeodata = true;
      worldSeries.geodata = am4geodata_worldLow;
      worldSeries.exclude = ["AQ"];

      var worldPolygon = worldSeries.mapPolygons.template;
      worldPolygon.tooltipText = "{name}";
      worldPolygon.nonScalingStroke = true;
      worldPolygon.strokeOpacity = 0.5;
      worldPolygon.fill = am4core.color("#eee");
      worldPolygon.propertyFields.fill = "color";

      var hs = worldPolygon.states.create("hover");
      hs.properties.fill = chart.colors.getIndex(9);


      // Create country specific series (but hide it for now)
      var countrySeries = chart.series.push(new am4maps.MapPolygonSeries());
      countrySeries.useGeodata = true;
      countrySeries.hide();
      countrySeries.geodataSource.events.on("done", function (ev) {
        worldSeries.hide();
        countrySeries.show();
      });

      var countryPolygon = countrySeries.mapPolygons.template;
      countryPolygon.tooltipText = "{name}";
      countryPolygon.nonScalingStroke = true;
      countryPolygon.strokeOpacity = 0.5;
      countryPolygon.fill = am4core.color("#eee");

      var hs = countryPolygon.states.create("hover");
      hs.properties.fill = chart.colors.getIndex(9);

      // Set up click events
      worldPolygon.events.on("hit", function (ev) {
        ev.target.series.chart.zoomToMapObject(ev.target);
        var map = ev.target.dataItem.dataContext.map;
        if (map) {
          ev.target.isHover = false;
          countrySeries.geodataSource.url = "https://www.amcharts.com/lib/4/geodata/json/" + map + ".json";
          countrySeries.geodataSource.load();
        }
      });

      // Set up data for countries
      var data = [];
      for (var id in am4geodata_data_countries2) {
        if (am4geodata_data_countries2.hasOwnProperty(id)) {
          var country = am4geodata_data_countries2[id];
          if (country.maps.length) {
            data.push({
              id: id,
              color: chart.colors.getIndex(continents[country.continent_code]),
              map: country.maps[0]
            });
          }
        }
      }
      worldSeries.data = data;

      // Zoom control
      chart.zoomControl = new am4maps.ZoomControl();

      var homeButton = new am4core.Button();
      homeButton.events.on("hit", function () {
        worldSeries.show();
        countrySeries.hide();
        chart.goHome();
      });

      homeButton.icon = new am4core.Sprite();
      homeButton.padding(7, 5, 7, 5);
      homeButton.width = 30;
      homeButton.icon.path = "M16,8 L14,8 L14,16 L10,16 L10,10 L6,10 L6,16 L2,16 L2,8 L0,8 L8,0 L16,8 Z M16,8";
      homeButton.marginBottom = 10;
      homeButton.parent = chart.zoomControl;
      homeButton.insertBefore(chart.zoomControl.plusButton);

      const body = document.querySelector('body')
      if (body.classList.contains('dark')) {
        amChartUpdate(chart, {
          dark: true
        })
      }

      document.addEventListener('ChangeColorMode', function (e) {
        amChartUpdate(chart, e.detail)
      })

    });
  }

  if (jQuery("#layout-1-chart-05").length) {
    const options = {
      series: [{
        name: 'Week',
        data: [80, 50, 30, 40, 100, 20],
      }, {
        name: 'Month',
        data: [20, 30, 40, 80, 20, 80],
      }, {
        name: 'Year',
        data: [44, 76, 78, 13, 43, 10],
      }],
      colors: ["#43d396", "#fe721c", "#876cfe"],
      chart: {
        height: 360,
        type: 'radar',
        dropShadow: {
          enabled: true,
          blur: 1,
          left: 1,
          top: 1
        }
      },

      stroke: {
        width: 2
      },
      fill: {
        opacity: 0.1
      },
      markers: {
        size: 0
      },
      xaxis: {
        categories: ['2011', '2012', '2013', '2014', '2015', '2016']
      }
    };

    const chart = new ApexCharts(document.querySelector("#layout-1-chart-05"), options);
    chart.render();
    const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })
  }

  if (jQuery("#layout-1-chart-06").length) {
  var options = {
          series: [{
          name: 'Total Likes',
          data: [86, 65, 96, 46, 30, 58,97]
        }, {
          name: 'Total Share',
          data: [55, 95, 45, 98, 55, 99,44]
        }],
          chart: {
          type: 'bar',
          height: 310
        },
    colors: ['#05bbc9','#876cfe'],

        plotOptions: {
          bar: {
            horizontal: false,
            columnWidth: '25%',
            endingShape: 'rounded'
          },
        },
        dataLabels: {
          enabled: false
        },
        stroke: {
          show: true,
          width: 2,
          colors: ['transparent']
        },
        xaxis: {
          categories: ['2020 Q1', '2020 Q2', '2020 Q3', '2020 Q4', '2020 Q5', '2020 Q6', '2020 Q7'],
        },
      yaxis: {
        show: true,
        labels: {
          minWidth: 20,
          maxWidth: 20
        }
      },
        fill: {
          opacity: 1
        },
        tooltip: {
          y: {
            formatter: function (val) {
              return "$ " + val + " thousands"
            }
          }
        }
        };

    var chart = new ApexCharts(document.querySelector("#layout-1-chart-06"), options);
        chart.render();
}
//layout-2
  if (jQuery('#layout-2-chart-01').length) {
    const options = {
      series: [{
        name: "Desktops",
        data: [10, 90, 40, 20, 35, 30, 45, 90, 30]
    }],
      chart: {
      height: 330,
      type: 'area',
      zoom: {
        enabled: false
      }
    },
    colors: ['#4788ff'],
    dataLabels: {
      enabled: false
    },
    stroke: {
      curve: 'smooth'
    },
    title: {
      text: 'Sales Trends by Month',
      align: 'left'
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
        opacity: 0.5
      },
    },
    xaxis: {
      categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
    }
    };

    const chart = new ApexCharts(document.querySelector("#layout-2-chart-01"), options);
    chart.render();
    const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })
  }
  if (jQuery("#layout-2-chart-02").length) {
    var options = {
      series: [{
        name: 'Top',
        data: [30, 60, 30, 60, 30, 60, 30]

      }, {
        name: 'New',
        data: [60, 30, 60, 30, 60, 30, 60]

      }],
      colors: ["#4788ff", "#f75676"],
      chart: {
        height: 330,
        type: 'line',
        zoom: {
          enabled: false
        },
        sparkline: {
          enabled: false,
        }
      },
      dataLabels: {
        enabled: false
      },
      stroke: {
        curve: 'smooth',
        width: 3
      },
      title: {
        text: '',
        align: 'left'
      },
      fill: {
        opacity: 1
      },
      grid: {
        row: {
          colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
          opacity: 0.5,

        },

      },
      xaxis: {
        categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        labels: {
          minHeight: 22,
          maxHeight: 35,
        }
      },
      yaxis: {
        labels: {
          offsetY: 0,
          minWidth: 15,
          maxWidth: 15,
        }
      },
      legend: {
        position: 'top',
        offsetX: -20        
      },
    };

    var chart = new ApexCharts(document.querySelector("#layout-2-chart-02"), options);
    chart.render();
    const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })

  }
  if (jQuery('#layout-2-chart-03').length) {
    Highcharts.chart('layout-2-chart-03', {
      chart: {
        plotBackgroundColor: null,
        plotBorderWidth: null,
        plotShadow: false,
        height: 300,
        type: 'pie'
      },
      title: {
        text: ''
      },
      tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
      },
      accessibility: {
        point: {
          valueSuffix: '%'
        }
      },
      plotOptions: {
        pie: {
          allowPointSelect: true,
          cursor: 'pointer',
          dataLabels: {
            enabled: false
          },
          showInLegend: true
        }
      },
      colors: ["#fe721c", "#05bbc9", "#876cfe"],
      series: [{
        name: 'Brands',
        colorByPoint: true,
        data: [{
          name: 'Desktop',
          y: 30,
          sliced: true,
          selected: true
        }, {
            name: 'Mobile',
          y: 40
        },{
            name: 'Tablet',
          y: 30
        }]
      }]
    });
    const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })
  }
  if (jQuery('#layout-2-chart-04').length) {
    am4core.ready(function () {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      var continents = {
        "AF": 0,
        "AN": 1,
        "AS": 2,
        "EU": 3,
        "NA": 4,
        "OC": 5,
        "SA": 6
      }

      // Create map instance
      var chart = am4core.create("layout-2-chart-04", am4maps.MapChart);
      chart.projection = new am4maps.projections.Miller();

      // Create map polygon series for world map
      var worldSeries = chart.series.push(new am4maps.MapPolygonSeries());
      worldSeries.useGeodata = true;
      worldSeries.geodata = am4geodata_worldLow;
      worldSeries.exclude = ["AQ"];

      var worldPolygon = worldSeries.mapPolygons.template;
      worldPolygon.tooltipText = "{name}";
      worldPolygon.nonScalingStroke = true;
      worldPolygon.strokeOpacity = 0.5;
      worldPolygon.fill = am4core.color("#eee");
      worldPolygon.propertyFields.fill = "color";

      var hs = worldPolygon.states.create("hover");
      hs.properties.fill = chart.colors.getIndex(9);


      // Create country specific series (but hide it for now)
      var countrySeries = chart.series.push(new am4maps.MapPolygonSeries());
      countrySeries.useGeodata = true;
      countrySeries.hide();
      countrySeries.geodataSource.events.on("done", function (ev) {
        worldSeries.hide();
        countrySeries.show();
      });

      var countryPolygon = countrySeries.mapPolygons.template;
      countryPolygon.tooltipText = "{name}";
      countryPolygon.nonScalingStroke = true;
      countryPolygon.strokeOpacity = 0.5;
      countryPolygon.fill = am4core.color("#eee");

      var hs = countryPolygon.states.create("hover");
      hs.properties.fill = chart.colors.getIndex(9);

      // Set up click events
      worldPolygon.events.on("hit", function (ev) {
        ev.target.series.chart.zoomToMapObject(ev.target);
        var map = ev.target.dataItem.dataContext.map;
        if (map) {
          ev.target.isHover = false;
          countrySeries.geodataSource.url = "https://www.amcharts.com/lib/4/geodata/json/" + map + ".json";
          countrySeries.geodataSource.load();
        }
      });

      // Set up data for countries
      var data = [];
      for (var id in am4geodata_data_countries2) {
        if (am4geodata_data_countries2.hasOwnProperty(id)) {
          var country = am4geodata_data_countries2[id];
          if (country.maps.length) {
            data.push({
              id: id,
              color: chart.colors.getIndex(continents[country.continent_code]),
              map: country.maps[0]
            });
          }
        }
      }
      worldSeries.data = data;

      // Zoom control
      chart.zoomControl = new am4maps.ZoomControl();

      var homeButton = new am4core.Button();
      homeButton.events.on("hit", function () {
        worldSeries.show();
        countrySeries.hide();
        chart.goHome();
      });

      homeButton.icon = new am4core.Sprite();
      homeButton.padding(7, 5, 7, 5);
      homeButton.width = 30;
      homeButton.icon.path = "M16,8 L14,8 L14,16 L10,16 L10,10 L6,10 L6,16 L2,16 L2,8 L0,8 L8,0 L16,8 Z M16,8";
      homeButton.marginBottom = 10;
      homeButton.parent = chart.zoomControl;
      homeButton.insertBefore(chart.zoomControl.plusButton);

      const body = document.querySelector('body')
      if (body.classList.contains('dark')) {
        amChartUpdate(chart, {
          dark: true
        })
      }

      document.addEventListener('ChangeColorMode', function (e) {
        amChartUpdate(chart, e.detail)
      })

    });
  }
  //layout-3

  if (jQuery("#layout-3-chart-01").length) {
    am4core.ready(function () {

      // Themes begin
      am4core.useTheme(am4themes_animated);
      // Themes end

      // Create map instance
      var chart = am4core.create("layout-3-chart-01", am4maps.MapChart);
      // Set map definition
      chart.geodata = am4geodata_worldLow;

      // Set projection
      chart.projection = new am4maps.projections.Miller();

      // Create map polygon series
      var polygonSeries = chart.series.push(new am4maps.MapPolygonSeries());

      // Exclude Antartica
      polygonSeries.exclude = ["AQ"];

      // Make map load polygon (like country names) data from GeoJSON
      polygonSeries.useGeodata = true;

      // Configure series
      var polygonTemplate = polygonSeries.mapPolygons.template;
      polygonTemplate.fill = '#585858';

      polygonTemplate.tooltipText = "{name}";
      polygonTemplate.polygon.fillOpacity = 0.6;


      // Create hover state and set alternative fill color
      var hs = polygonTemplate.states.create("hover");
      hs.properties.fill = chart.colors.getIndex(0);

      // Add image series
      var imageSeries = chart.series.push(new am4maps.MapImageSeries());
      imageSeries.mapImages.template.propertyFields.longitude = "longitude";
      imageSeries.mapImages.template.propertyFields.latitude = "latitude";
      imageSeries.mapImages.template.tooltipText = "{title}";
      imageSeries.mapImages.template.propertyFields.url = "url";

      var circle = imageSeries.mapImages.template.createChild(am4core.Circle);
      circle.radius = 3;
      circle.propertyFields.fill = "color";

      var circle2 = imageSeries.mapImages.template.createChild(am4core.Circle);
      circle2.radius = 3;
      circle2.propertyFields.fill = "color";


      circle2.events.on("inited", function (event) {
        animateBullet(event.target);
      })


      function animateBullet(circle) {
        var animation = circle.animate([{ property: "scale", from: 1, to: 5 }, { property: "opacity", from: 1, to: 0 }], 1000, am4core.ease.circleOut);
        animation.events.on("animationended", function (event) {
          animateBullet(event.target.object);
        })
      }

      var colorSet = new am4core.ColorSet();

      imageSeries.data = [{
        "title": "Brussels",
        "latitude": 50.8371,
        "longitude": 4.3676,
        "color": colorSet.next()
      }, {
        "title": "Copenhagen",
        "latitude": 55.6763,
        "longitude": 12.5681,
        "color": colorSet.next()
      }, {
        "title": "Paris",
        "latitude": 48.8567,
        "longitude": 2.3510,
        "color": colorSet.next()
      }, {
        "title": "Reykjavik",
        "latitude": 64.1353,
        "longitude": -21.8952,
        "color": colorSet.next()
      }, {
        "title": "Moscow",
        "latitude": 55.7558,
        "longitude": 37.6176,
        "color": colorSet.next()
      }, {
        "title": "Madrid",
        "latitude": 40.4167,
        "longitude": -3.7033,
        "color": colorSet.next()
      }, {
        "title": "London",
        "latitude": 51.5002,
        "longitude": -0.1262,
        "url": "http://www.google.co.uk",
        "color": colorSet.next()
      }, {
        "title": "Peking",
        "latitude": 39.9056,
        "longitude": 116.3958,
        "color": colorSet.next()
      }, {
        "title": "New Delhi",
        "latitude": 28.6353,
        "longitude": 77.2250,
        "color": colorSet.next()
      }, {
        "title": "Tokyo",
        "latitude": 35.6785,
        "longitude": 139.6823,
        "url": "http://www.google.co.jp",
        "color": colorSet.next()
      }, {
        "title": "Ankara",
        "latitude": 39.9439,
        "longitude": 32.8560,
        "color": colorSet.next()
      }, {
        "title": "Buenos Aires",
        "latitude": -34.6118,
        "longitude": -58.4173,
        "color": colorSet.next()
      }, {
        "title": "Brasilia",
        "latitude": -15.7801,
        "longitude": -47.9292,
        "color": colorSet.next()
      }, {
        "title": "Ottawa",
        "latitude": 45.4235,
        "longitude": -75.6979,
        "color": colorSet.next()
      }, {
        "title": "Washington",
        "latitude": 38.8921,
        "longitude": -77.0241,
        "color": colorSet.next()
      }, {
        "title": "Kinshasa",
        "latitude": -4.3369,
        "longitude": 15.3271,
        "color": colorSet.next()
      }, {
        "title": "Cairo",
        "latitude": 30.0571,
        "longitude": 31.2272,
        "color": colorSet.next()
      }, {
        "title": "Pretoria",
        "latitude": -25.7463,
        "longitude": 28.1876,
        "color": colorSet.next()
      }];

      const body = document.querySelector('body')
      if (body.classList.contains('dark')) {
        amChartUpdate(chart, {
          dark: true
        })
      }

      document.addEventListener('ChangeColorMode', function (e) {
        amChartUpdate(chart, e.detail)
      })

    });
  }

  if (jQuery('#layout-3-chart-02').length) {
    Highcharts.chart('layout-3-chart-02', {
      chart: {
        plotBackgroundColor: null,
        plotBorderWidth: null,
        plotShadow: false,
        height: 300,
        type: 'pie'
      },
      title: {
        text: ''
      },
      tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
      },
      accessibility: {
        point: {
          valueSuffix: '%'
        }
      },
      plotOptions: {
        pie: {
          allowPointSelect: true,
          cursor: 'pointer',
          dataLabels: {
            enabled: false
          },
          showInLegend: true
        }
      },
      colors: ["#876cfe", "#05bbc9"],
      series: [{
        name: 'Brands',
        colorByPoint: true,
        data: [{
          name: 'Man : 3,272978',
          y: 80,
          sliced: true,
          selected: true
        }, {
            name: 'Woman : 83,272978',
          y: 30
        }]
      }]
    });
  }
  if (jQuery("#layout-3-chart-03").length) {
    const options = {
      series: [{
        name: 'Frauds',
        data: [53, 55, 45, 40, 40, 28, 35, 25, 2]
      }, {
          name: 'Paid Clicks',
        data: [63, 62, 52, 72, 55, 80, 70, 50, 60]
      }, {
          name: 'Total Clicks',
        data: [150, 135, 144, 115, 120, 114, 133, 124, 100]
      }],
      colors: ['#05bbc9', '#ffca44', '#876cfe'],
      labels: ['20 Feb', '21 Feb', '22 Feb', '23 Feb', '24 Feb', '25 Feb', '25 Feb', '26 Feb', '27 Feb', '28 Feb' ],
      chart: {
        type: 'bar',
        height: 330,
        stacked: true,
        zoom: {
          enabled: true
        }
      },
      responsive: [{
        breakpoint: 480,
        options: {
          legend: {
            position: 'bottom',
            offsetX: -10,
            offsetY: 0
          }
        }
      }],
      plotOptions: {
        bar: {
          horizontal: false,
          columnWidth: '12%',
          endingShape: 'rounded'
        },
      },
      xaxis: {
        show: true
        
      },
      yaxis: {
        show: true,
        labels: {
          minWidth: 20,
          maxWidth: 20
        }
      },
      legend: {
        position: 'bottom',
        offsetX: 0,
        offsetY: -10

      },
      fill: {
        opacity: 1
      },
      dataLabels: {
        enabled: false
      }
      
    };
    const chart = new ApexCharts(document.querySelector("#layout-3-chart-03"), options);
    chart.render();
    const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })
  }
  
  if (jQuery("#layout-3-chart-04").length) {
    var options = {
      series: [86, 40, 53, 63],
      chart: {
        height: 350,
        type: 'radialBar',
      },
      plotOptions: {
        radialBar: {
          dataLabels: {
            name: {
              fontSize: '22px',
            },
            value: {
              fontSize: '16px',
            },
            total: {
              show: true,
              label: 'Total',
              formatter: function (w) {
                // By default this function returns the average of all series. The below is just an example to show the use of custom formatter function
                return 249
              }
            }
          }
        }
      },
      colors: ['#876cfe', '#4788ff', '#ffca44', '#fe721c'],
      labels: ['Picture view', 'Comments', 'Video plays', 'Impressions'],
    };

    var chart = new ApexCharts(document.querySelector("#layout-3-chart-04"), options);
    chart.render();
    const body = document.querySelector('body')
    if (body.classList.contains('dark')) {
      apexChartUpdate(chart, {
        dark: true
      })
    }

    document.addEventListener('ChangeColorMode', function (e) {
      apexChartUpdate(chart, e.detail)
    })

  }
})(jQuery);


