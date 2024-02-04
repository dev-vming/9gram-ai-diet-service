import { Controller, Get, Query, Req, UseGuards } from "@nestjs/common";
import { ApiOperation, ApiTags } from "@nestjs/swagger";
import { CumulativeRecordService } from "./cumulative-record.service";
import { isLoggedInGuard } from "src/auth/utils/guards/isLoggedin.guard";

@Controller("cumulative-record")
@ApiTags("Cumulative Record API")
export class CumulativeRecordController {
  constructor(private cumulativeRecordService: CumulativeRecordService) {}

  @Get("/")
  @UseGuards(isLoggedInGuard)
  @ApiOperation({
    summary: "월별/일별 누적 식단 데이터 조회하기",
    description:
      "유저의 누적 식단 데이터를 조회한다. Query는 date이거나 month이다.",
  })
  public async getRecord(
    @Req() request: any,
    @Query("date") date: Date,
    @Query("month") month: Date
  ) {
    try {
      const userId = request.user.userId;

      // 1) [Cumulative Table] 유저의 일별 모든 meal type의 칼로리 합산 -> totalCalories
      // 2) [Cumulative Table] 유저의 일별 모든 meal type의 탄단지 합산 -> totalNutrient
      // 3) [HealthInfo Table] 유저의 목표 칼로리 조회 -> targetCalories
      // 4) [HealthInfo Table] 유저의 목표 영양성분 조회 -> recommendNutrient
      // 5) [Cumulative Table & Image Table] meal type별 칼로리와 image -> dateArr
      if (date) {
        const result = await this.cumulativeRecordService.getDateRecord(
          date,
          userId
        );
        return result;
      }

      if (month) {
        const data = await this.cumulativeRecordService.getMonthRecord(
          month,
          userId
        );
        const dateArr = data.map((item) => item.date.getDate());
        const caloriesArr = data.map((item) => item.mealTotalCalories);

        return {
          existedDate: dateArr,
          totalCalData: caloriesArr,
        };
      }
    } catch (error) {
      throw error;
    }
  }

  @Get("/meal")
  @UseGuards(isLoggedInGuard)
  @ApiOperation({
    summary: "일별 & meal 타입별 누적 식단 데이터 조회하기",
    description: "유저의 누적 식단 데이터를 조회한다.",
  })
  public async getDateMealTypeRecord(
    @Req() request: any,
    @Query("date") date: Date
  ) {
    try {
      const userId = request.user.userId;
      const sortedDateArr =
        await this.cumulativeRecordService.transformMealTypeRecord(
          date,
          userId
        );
      return { dateArr: sortedDateArr };
    } catch (error) {
      throw error;
    }
  }

  @Get("/month")
  @UseGuards(isLoggedInGuard)
  @ApiOperation({
    summary: "월별 & meal 타입별 누적 식단 데이터 조회하기",
    description: "유저의 월별 아침, 점심, 저녁 누적 식단 데이터를 조회한다.",
  })
  public async getMonthDetailRecord(
    @Req() request: any,
    @Query("month") month: Date,
    @Query("page") page: number
  ) {
    try {
      const userId = request.user.userId;
      const transformedData =
        await this.cumulativeRecordService.getMonthDetailRecord(
          month,
          page,
          userId
        );
      return transformedData;
    } catch (error) {
      throw error;
    }
  }
}
