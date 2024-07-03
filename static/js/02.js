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

    if (jQuery("#hospital-chart-02").length) var donut = new Morris.Donut({
        element: "hospital-chart-02",
        resize: !0,
        colors: ["#4788ff", "#4fd69c", "#37d5f2", "#f75676"],
        data: [
          {
            label: "New Patient",
            value: 40
          },
          {
            label: "Exit Patient",
            value: 12
          },
          {
            label: "ICU Patient",
            value: 20
          },
          {
            label: "Discharge Patient",
            value: 33
          }
        ],
        hideHover: "auto"
    });

    if (jQuery("#hospital-chart-03").length) {
        const options = {
          series: [{
            name: 'Operation',
            data: [44, 25, 10, 60, 50, 25, 40, 20]
          }, {
            name: 'Visitors',
            data: [25, 38, 70, 0, 30, 15, 30, 30]
          }],
          colors: ['#4788ff', '#4fd69c'],
          chart: {
            type: 'bar',
            height: 340,
            stacked: true,
            zoom: {
              enabled: true
            }
          },
          responsive: [{
            breakpoint: 580,
            options: {
              legend: {
                position: 'bottom',
                offsetX: -30,
                offsetY: 0,
              }
            }
          }],
          plotOptions: {
            bar: {
              horizontal: false,
              borderRadius: 4
            },
          },
          xaxis: {
            type: 'category',
            categories: ['India', 'Canada', 'U.S.A', 'Africa', 'London ', 'Europe ','Dubai','Kuvet'],
          },
           yaxis: {
            labels: {
                offsetY: 0,
                minWidth: 20,
                maxWidth: 20,
              }
           },
          legend: {
            position: 'top',
            offsetX: -35
          },
          fill: {
            opacity: 1
          },
          dataLabels: {
            enabled: false
          }
        };
      
        const chart = new ApexCharts(document.querySelector("#hospital-chart-03"), options);
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

      if (jQuery("#dash-chart-04").length) {
        const options = {
          series: [{
            name: 'Success',
            data: [110, 85, 87, 40, 45, 20, 91, 45, 94, 88]
          }, {
            name: 'Failed',
            data: [40, 55, 35, 30, 75, 80, 63, 45, 66, 115]
          }],
          chart: {
            type: 'bar',
            height: 280
          },
          colors: ['#4788ff', '#f75676'],
          plotOptions: {
            bar: {
              horizontal: false,
              borderRadius: 4
            },
          },
          legend: {
            position: 'bottom',
            offsetX: 35
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
            categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
          },
          yaxis: {
            title: {
              text: '$ (thousands)'
            },
            labels: {
                offsetY: 0,
                minWidth: 20,
                maxWidth: 20,
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
      
        const chart = new ApexCharts(document.querySelector("#dash-chart-04"), options);
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

      if (jQuery("#hospital-chart-05").length) {
        var options = {
          series: [22, 83, 10, 30],
          colors: ['#f75676', '#4fd69c', '#37d5f2', '#4788ff'],
          chart: {
            height: 290,
            type: 'pie',
          },
          legend: {
            position: 'bottom',
            offsetX: 0
          },
          labels: ['Cancelled', 'Completed', 'Pending', 'Upcoming'],
      
        };
        var chart = new ApexCharts(document.querySelector("#hospital-chart-05"), options);
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