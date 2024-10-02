'use client'

import React from 'react';
import { useState, useEffect } from 'react'
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  PieChart, Pie, Cell, LineChart, Line, ResponsiveContainer 
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Download, BarChart2, PieChartIcon, TrendingUp, Scissors } from "lucide-react"

const initialDeepfakeCases = [
  { name: 'Reported', value: 400 },
  { name: 'Solved', value: 300 },
]

const initialPlatformData = [
  { name: 'Facebook', value: 400 },
  { name: 'Twitter', value: 300 },
  { name: 'Instagram', value: 200 },
  { name: 'TikTok', value: 100 },
]

const initialMonthlyData = [
  { name: 'Jan', blocked: 40 },
  { name: 'Feb', blocked: 35 },
  { name: 'Mar', blocked: 55 },
  { name: 'Apr', blocked: 60 },
  { name: 'May', blocked: 45 },
  { name: 'Jun', blocked: 48 },
  { name: 'Jul', blocked: 30 },
]

const initialManipulationData = [
  { name: 'Face Swapping', value: 35 },
  { name: 'Voice Cloning', value: 25 },
  { name: 'Lip Syncing', value: 20 },
  { name: 'Body Manipulation', value: 15 },
  { name: 'Other', value: 5 },
]

const initialTableData = [
  { id: 1, date: '2023-06-01', type: 'Video', platform: 'TikTok', status: 'Fake', confidence: 0.92, nature: 'Face swapping', technology: 'GANs' },
  { id: 2, date: '2023-06-02', type: 'Image', platform: 'Instagram', status: 'Real', confidence: 0.88, nature: 'N/A', technology: 'N/A' },
  { id: 3, date: '2023-06-03', type: 'Audio', platform: 'Twitter', status: 'Fake', confidence: 0.95, nature: 'Voice cloning', technology: 'WaveNet' },
  { id: 4, date: '2023-06-04', type: 'Video', platform: 'Facebook', status: 'Fake', confidence: 0.89, nature: 'Lip syncing', technology: 'AutoEncoder' },
  { id: 5, date: '2023-06-05', type: 'Image', platform: 'Instagram', status: 'Real', confidence: 0.91, nature: 'N/A', technology: 'N/A' },
]

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8']

export default function Dashboard() {
  const [timeRange, setTimeRange] = useState('7d')
  const [platform, setPlatform] = useState('all')
  const [mediaType, setMediaType] = useState('all')
  const [deepfakeCases, setDeepfakeCases] = useState(initialDeepfakeCases)
  const [platformData, setPlatformData] = useState(initialPlatformData)
  const [monthlyData, setMonthlyData] = useState(initialMonthlyData)
  const [manipulationData, setManipulationData] = useState(initialManipulationData)
  const [tableData, setTableData] = useState(initialTableData)

  useEffect(() => {
    // Simulating data filtering based on selected filters
    const filteredTableData = initialTableData.filter(item => 
      (platform === 'all' || item.platform === platform) &&
      (mediaType === 'all' || item.type === mediaType)
    )
    setTableData(filteredTableData)

    // Update chart data based on filters (simplified simulation)
    setDeepfakeCases(deepfakeCases.map(item => ({ ...item, value: item.value * Math.random() * 0.5 + 0.5 })))
    setPlatformData(platformData.map(item => ({ ...item, value: item.value * Math.random() * 0.5 + 0.5 })))
    setMonthlyData(monthlyData.map(item => ({ 
      ...item, 
      blocked: item.blocked * Math.random() * 0.5 + 0.5,
    })))
    setManipulationData(manipulationData.map(item => ({ ...item, value: item.value * Math.random() * 0.5 + 0.5 })))
  }, [timeRange, platform, mediaType])

  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
  
    return (
      <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central">
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  const downloadReport = (format: 'csv' | 'pdf') => {
    console.log(`Downloading ${format} report...`)
  }

  return (
    <div className="space-y-6 p-6 bg-gray-50">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h1 className="text-3xl font-bold text-gray-900">Deepfake Detection Dashboard</h1>
        <div className="flex flex-wrap gap-4">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 90 days</SelectItem>
            </SelectContent>
          </Select>
          <Select value={platform} onValueChange={setPlatform}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Platform" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Platforms</SelectItem>
              <SelectItem value="Facebook">Facebook</SelectItem>
              <SelectItem value="Twitter">Twitter</SelectItem>
              <SelectItem value="Instagram">Instagram</SelectItem>
              <SelectItem value="TikTok">TikTok</SelectItem>
            </SelectContent>
          </Select>
          <Select value={mediaType} onValueChange={setMediaType}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Media type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="Video">Video</SelectItem>
              <SelectItem value="Image">Image</SelectItem>
              <SelectItem value="Audio">Audio</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Deepfake Cases: Reported vs Solved</CardTitle>
            <BarChart2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={deepfakeCases}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Platform Distribution</CardTitle>
            <PieChartIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={platformData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={renderCustomizedLabel}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {platformData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Monthly Trend: Deepfakes Blocked</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={monthlyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="blocked" stroke="#8884d8" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Manipulation Techniques Distribution</CardTitle>
            <Scissors className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={manipulationData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#8884d8">
                  {manipulationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Deepfake Incidents</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[100px]">Date</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Platform</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Nature of Manipulation</TableHead>
                <TableHead>Technology Used</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {tableData.map((row) => (
                <TableRow key={row.id}>
                  <TableCell className="font-medium">{row.date}</TableCell>
                  <TableCell>{row.type}</TableCell>
                  <TableCell>{row.platform}</TableCell>
                  <TableCell>{row.status}</TableCell>
                  <TableCell>{row.confidence.toFixed(2)}</TableCell>
                  <TableCell>{row.nature}</TableCell>
                  <TableCell>{row.technology}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="flex justify-end space-x-4">
        <Button onClick={() => downloadReport('csv')} variant="outline">
          <Download className="mr-2 h-4 w-4" /> Download CSV
        </Button>
        <Button onClick={() => downloadReport('pdf')}>
          <Download className="mr-2 h-4 w-4" /> Download PDF
        </Button>
      </div>
    </div>
  )
}