import { Controller, Get, Param, Query, Post, Put, Delete, Body, Req, HttpStatus, Res, UsePipes, UseGuards } from "@nestjs/common";
import { ApiBody, ApiOperation, ApiTags } from "@nestjs/swagger";
import { RecordService } from "./record.service";
import { Record } from "./record.entity";
import { ValidateFoodPipe } from "./pipes/record.pipe";
import { RecordDto } from "./dtos/record.dto";
import { LocalAuthGuard } from "src/auth/utils/guards/local.auth.guard";
import { isLoggedInGuard } from "src/auth/utils/guards/isLoggedin.guard";
@Controller("records")
@ApiTags("Record API")
export class RecordController {
  constructor(private recordService: RecordService) {}

  // GET /records?date=yyyy-mm-dd
  // @ApiBody({ type: LocalLoginDto, description: '로그인 정보'})

  //@UseGuards(LocalAuthGuard)
  @UseGuards(isLoggedInGuard)
  @Get()
  @ApiOperation({ summary: "특정 날짜의 식단 조회" })
  async getDailyRecords(
    @Req() req,
    @Query("date") date: string
  ): Promise<Record[]> {
    console.log("req userId : ", req.user.userId)
    const userId = req.user.userId

    const records = await this.recordService.getDailyRecord(userId, date);
    return records;
  }

  // POST /records
  @UseGuards(isLoggedInGuard)
  @Post()
  @UsePipes(new ValidateFoodPipe())
  @ApiOperation({ summary: "식단 기록" })
  async createRecord(
    @Req() req,
    @Query('date') date: string,
    @Query('mealType') queryMealType: number,
    @Body() recordDto: RecordDto
  ): Promise<string> {
    const userId = req.user.userId
    await this.recordService.createRecord(userId, date, queryMealType, recordDto);
    return "식단 기록 성공";
  }


  // PUT /records?date=yyyy-mm-dd&mealType
  @UseGuards(isLoggedInGuard)
  @Put()
  @ApiOperation({ summary: "특정 날짜와 식단 구분 수정" })
  async updateDailyRecord(
    @Req() req,
    @Query('date') date: string,
    @Query('mealType') queryMealType: number,
    @Body() recordDto: RecordDto,
  ): Promise<string> {
    const userId = req.user.userId
    await this.recordService.updateDailyRecord(userId, date, queryMealType, recordDto);
    return "식단 수정 성공";
  }

  // DELETE /records?date=yyyy-mm-dd&mealType
  @UseGuards(isLoggedInGuard)
  @Delete()
  @ApiOperation({ summary: "특정 날짜와 식단 구분 삭제" })
  async deleteDailyRecord(
    @Req() req,
    @Query('date') date: string,
    @Query('mealType') mealType: number,
  ): Promise<string> {
    const userId = req.user.userId
    await this.recordService.deleteDailyRecord(userId, date, mealType);
    return "식단 삭제 성공"
  }
}